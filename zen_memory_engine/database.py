"""
database.py — FAISS + JSON Vector Storage Module
==================================================
This module manages persistent, local vector storage using:
  • FAISS  — for fast cosine-similarity search on dense vectors
  • JSON   — sidecar files that map FAISS indices back to raw text & metadata

Why not ChromaDB?
  ChromaDB ≤ 1.5 relies on pydantic v1 internally, which is broken on Python
  3.14+.  FAISS is a battle-tested alternative from Meta that has zero such
  dependency issues.

Key design decisions:
  • Each user (identified by email) gets their own FAISS index + JSON file.
    This guarantees strict data separation.
  • Data is persisted to disk (./faiss_data/<sanitised_email>/) so memories
    survive server restarts.

Usage:
    from database import save_memory, search_memories
    save_memory("user@example.com", "Hello world", [0.1, 0.2, ...])
    results = search_memories("user@example.com", query_vector, limit=5)
"""

import os
import re
import json
import uuid
import numpy as np
import faiss

# ---------------------------------------------------------------------------
# Storage directory
# ---------------------------------------------------------------------------
FAISS_DATA_DIR = "./faiss_data"
os.makedirs(FAISS_DATA_DIR, exist_ok=True)

# Dimension of the all-MiniLM-L6-v2 embeddings
VECTOR_DIM = 384

print(f"[database] FAISS storage initialised — data directory: {FAISS_DATA_DIR}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _sanitise_name(email: str) -> str:
    """
    Convert an email into a filesystem-safe directory name.
    Replaces all non-alphanumeric characters with underscores.
    """
    safe = re.sub(r"[^a-zA-Z0-9]", "_", email)
    return safe.strip("_") or "unknown"


def _user_dir(email: str) -> str:
    """Return (and create if needed) the storage directory for a user."""
    path = os.path.join(FAISS_DATA_DIR, _sanitise_name(email))
    os.makedirs(path, exist_ok=True)
    return path


def _index_path(email: str) -> str:
    """Path to the user's FAISS index file."""
    return os.path.join(_user_dir(email), "index.faiss")


def _meta_path(email: str) -> str:
    """Path to the user's metadata JSON file (maps index → text)."""
    return os.path.join(_user_dir(email), "meta.json")


def _load_index(email: str):
    """
    Load an existing FAISS index from disk, or create a new one.
    We use IndexFlatIP (inner-product) on L2-normalised vectors,
    which is equivalent to cosine similarity.
    """
    path = _index_path(email)
    if os.path.exists(path):
        return faiss.read_index(path)
    else:
        # IndexFlatIP = brute-force inner-product search (cosine on normed vecs)
        return faiss.IndexFlatIP(VECTOR_DIM)


def _load_meta(email: str) -> list[dict]:
    """Load the metadata list from JSON, or return an empty list."""
    path = _meta_path(email)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_index(email: str, index):
    """Persist the FAISS index to disk."""
    faiss.write_index(index, _index_path(email))


def _save_meta(email: str, meta: list[dict]):
    """Persist the metadata list to disk as JSON."""
    with open(_meta_path(email), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def save_memory(email: str, text: str, vector: list[float]) -> str:
    """
    Persist a piece of text and its vector embedding for a specific user.

    Parameters
    ----------
    email : str
        The user's email address (used to isolate their data).
    text : str
        The raw chat message or snippet to store.
    vector : list[float]
        The 384-dimensional embedding produced by embedder.text_to_vector().

    Returns
    -------
    str
        A unique ID assigned to the stored memory.
    """
    # Load existing data.
    index = _load_index(email)
    meta = _load_meta(email)

    # Generate a unique ID for this memory entry.
    memory_id = str(uuid.uuid4())

    # Convert vector to a numpy array and L2-normalise it so that
    # inner-product search is equivalent to cosine similarity.
    vec = np.array([vector], dtype="float32")
    faiss.normalize_L2(vec)

    # Add the vector to the FAISS index.
    index.add(vec)

    # Record the text + ID in the metadata sidecar.
    meta.append({"id": memory_id, "text": text, "email": email})

    # Persist to disk.
    _save_index(email, index)
    _save_meta(email, meta)

    print(f"[database] Saved memory {memory_id} for {email}")
    return memory_id


def search_memories(
    email: str,
    query_vector: list[float],
    limit: int = 5,
) -> list[str]:
    """
    Perform a semantic (cosine-similarity) search over a user's stored
    memories and return the most relevant text snippets.

    Parameters
    ----------
    email : str
        The user's email address — search is scoped to their data only.
    query_vector : list[float]
        The embedding of the query/question to search with.
    limit : int, optional
        Maximum number of results to return (default 5).

    Returns
    -------
    list[str]
        A list of the most relevant past text snippets, ordered by semantic
        similarity (closest first).  Returns an empty list if the user has
        no stored memories yet.
    """
    index = _load_index(email)
    meta = _load_meta(email)

    # If the user has no memories yet, return early.
    if index.ntotal == 0:
        print(f"[database] No memories found for {email}")
        return []

    # Clamp the limit so we never ask for more results than exist.
    effective_limit = min(limit, index.ntotal)

    # Normalise the query vector for cosine similarity.
    q = np.array([query_vector], dtype="float32")
    faiss.normalize_L2(q)

    # Search — returns (distances, indices) arrays.
    distances, indices = index.search(q, effective_limit)

    # Map FAISS indices back to original text via the metadata list.
    documents = []
    for idx in indices[0]:
        if 0 <= idx < len(meta):
            documents.append(meta[idx]["text"])

    print(f"[database] Found {len(documents)} memories for {email}")
    return documents
