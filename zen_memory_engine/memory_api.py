"""
memory_api.py — FastAPI Server for the Memory Engine
=====================================================
This is the entry-point for the microservice. It exposes two POST endpoints:

  POST /remember   — store a new memory for a user
  POST /recall     — retrieve relevant past memories for a user

Data flow:
  1. Client sends JSON  →  FastAPI validates it with Pydantic models
  2. The text is converted to a vector via  embedder.text_to_vector()
  3. The vector + text are saved / searched via  database.save_memory()
     or database.search_memories()
  4. A JSON response is returned to the client

Run with:
    uvicorn memory_api:app --reload --port 8100
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr

# Import our internal modules — they live in the same package.
from embedder import text_to_vector
from database import save_memory, search_memories

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Zen Memory Engine",
    description="Long-Term Memory microservice — store and recall user chat "
                "history as semantic vector embeddings.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response Schemas (Pydantic models)
# ---------------------------------------------------------------------------
class RememberRequest(BaseModel):
    """Schema for the /remember endpoint."""
    email: EmailStr          # validated email address
    text: str                # the chat message or snippet to store


class RememberResponse(BaseModel):
    """Schema for the /remember response."""
    status: str
    memory_id: str


class RecallRequest(BaseModel):
    """Schema for the /recall endpoint."""
    email: EmailStr          # whose memories to search
    query_text: str          # the question / prompt to search with


class RecallResponse(BaseModel):
    """Schema for the /recall response."""
    email: str
    query_text: str
    memories: list[str]      # the retrieved text snippets


# ---------------------------------------------------------------------------
# Health-check endpoint
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def health_check():
    """Simple liveness probe — useful for monitoring / integration tests."""
    return {"status": "ok", "service": "Zen Memory Engine"}


# ---------------------------------------------------------------------------
# POST /remember — store a new memory
# ---------------------------------------------------------------------------
@app.post("/remember", response_model=RememberResponse, tags=["Memory"])
async def remember(request: RememberRequest):
    """
    Accept a piece of text from a user and store it as a vector embedding.

    Steps:
      1. Embed the text  →  384-dim vector
      2. Save the text + vector in the user's ChromaDB collection
      3. Return the unique memory ID
    """
    try:
        # Step 1 — Convert text to vector.
        vector = text_to_vector(request.text)

        # Step 2 — Persist to ChromaDB.
        memory_id = save_memory(
            email=request.email,
            text=request.text,
            vector=vector,
        )

        # Step 3 — Respond with success.
        return RememberResponse(
            status="saved",
            memory_id=memory_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /recall — retrieve relevant memories
# ---------------------------------------------------------------------------
@app.post("/recall", response_model=RecallResponse, tags=["Memory"])
async def recall(request: RecallRequest):
    """
    Search a user's stored memories for snippets semantically similar to
    the provided query text.

    Steps:
      1. Embed the query text  →  384-dim vector
      2. Cosine-similarity search against the user's ChromaDB collection
      3. Return the top-N matching text snippets
    """
    try:
        # Step 1 — Convert the query to a vector.
        query_vector = text_to_vector(request.query_text)

        # Step 2 — Search the user's memory store.
        memories = search_memories(
            email=request.email,
            query_vector=query_vector,
            limit=5,
        )

        # Step 3 — Return the results.
        return RecallResponse(
            email=request.email,
            query_text=request.query_text,
            memories=memories,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Direct execution (alternative to `uvicorn memory_api:app`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
