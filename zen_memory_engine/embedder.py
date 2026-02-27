"""
embedder.py — Text-to-Vector Embedding Module
================================================
This module initialises the sentence-transformers model and exposes a single
utility function that converts raw text into a dense vector embedding.

Model used: all-MiniLM-L6-v2
  • Produces 384-dimensional vectors
  • Fast, lightweight, and runs 100 % locally (no API keys needed)
  • Great quality for semantic-similarity tasks

Usage:
    from embedder import text_to_vector
    vec = text_to_vector("Hello, how are you?")   # returns list[float]
"""

from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Model Initialisation (singleton — loaded once when the module is imported)
# ---------------------------------------------------------------------------
# The model is downloaded on first run (~80 MB) and cached locally by the
# sentence-transformers library.  Subsequent imports reuse the cached copy.
MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(MODEL_NAME)

print(f"[embedder] Loaded model: {MODEL_NAME}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def text_to_vector(text: str) -> list[float]:
    """
    Convert a string of text into a 384-dimensional vector embedding.

    Parameters
    ----------
    text : str
        The raw text to embed (a chat message, a query, etc.).

    Returns
    -------
    list[float]
        A list of 384 floating-point numbers representing the semantic
        meaning of the input text.
    """
    # .encode() returns a numpy array; .tolist() converts it to a plain
    # Python list so it can be JSON-serialised and stored in ChromaDB.
    embedding = _model.encode(text)
    return embedding.tolist()
