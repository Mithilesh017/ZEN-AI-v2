"""
database.py — Pinecone Vector Storage Module
==============================================
This module manages cloud vector storage using Pinecone (v3 client).

Why Pinecone instead of FAISS?
  • FAISS stores data on disk — Render's free tier has ephemeral storage
    that gets wiped on every deploy/restart.
  • Pinecone is a managed cloud vector DB with a generous free tier
    (100K vectors) — data persists permanently.

Key design decisions:
  • All users share a single Pinecone index ("zen-memory").
  • User isolation is achieved via metadata filtering on the email field.
  • Each vector stores: id, embedding, metadata = {email, text}.

Required env vars:
  PINECONE_API_KEY — your Pinecone API key (free at app.pinecone.io)

Usage:
    from database import save_memory, search_memories
    save_memory("user@example.com", "Hello world", [0.1, 0.2, ...])
    results = search_memories("user@example.com", query_vector, limit=5)
"""

import os
import uuid
from pinecone import Pinecone

# ---------------------------------------------------------------------------
# Pinecone Client Initialisation
# ---------------------------------------------------------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to the pre-created index.
# You must create this index in the Pinecone dashboard first:
#   Name: zen-memory | Dimensions: 384 | Metric: cosine
INDEX_NAME = "chatbot-memory"
index = pc.Index(INDEX_NAME)

print(f"[database] Connected to Pinecone index: {INDEX_NAME}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def save_memory(email: str, text: str, vector: list[float]) -> str:
    """
    Persist a piece of text and its vector embedding for a specific user
    in Pinecone.

    Parameters
    ----------
    email : str
        The user's email address (stored as metadata for filtering).
    text : str
        The raw chat message or snippet to store.
    vector : list[float]
        The 384-dimensional embedding produced by embedder.text_to_vector().

    Returns
    -------
    str
        A unique ID assigned to the stored memory.
    """
    # Generate a unique ID for this memory entry.
    memory_id = str(uuid.uuid4())

    # Upsert the vector with metadata into Pinecone.
    index.upsert(
        vectors=[
            {
                "id": memory_id,
                "values": vector,
                "metadata": {
                    "email": email,
                    "text": text,
                },
            }
        ]
    )

    print(f"[database] Saved memory {memory_id} for {email}")
    return memory_id


def search_memories(
    email: str,
    query_vector: list[float],
    limit: int = 5,
) -> list[str]:
    """
    Perform a semantic (cosine-similarity) search over a user's stored
    memories in Pinecone and return the most relevant text snippets.

    Parameters
    ----------
    email : str
        The user's email address — results are filtered to this user only.
    query_vector : list[float]
        The embedding of the query/question to search with.
    limit : int, optional
        Maximum number of results to return (default 5).

    Returns
    -------
    list[str]
        A list of the most relevant past text snippets, ordered by semantic
        similarity (closest first).  Returns an empty list if no memories
        are found.
    """
    # Query Pinecone with a metadata filter so we only search this user's data.
    results = index.query(
        vector=query_vector,
        top_k=limit,
        include_metadata=True,
        filter={"email": {"$eq": email}},
    )

    # Extract the text from each match's metadata.
    documents = []
    for match in results.get("matches", []):
        text = match.get("metadata", {}).get("text", "")
        if text:
            documents.append(text)

    print(f"[database] Found {len(documents)} memories for {email}")
    return documents
