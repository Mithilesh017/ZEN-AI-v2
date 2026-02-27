"""
embedder.py — HuggingFace Inference API Embedding Module
==========================================================
This module calls the HuggingFace Inference API to convert text into
384-dimensional vector embeddings using the all-MiniLM-L6-v2 model.

Why cloud instead of local?
  • sentence-transformers + PyTorch need ~1.5 GB RAM — too much for
    Render's free tier (512 MB).
  • The HF Inference API runs the same model on their servers for free.

Required env var:
  HF_TOKEN — your HuggingFace access token (free at huggingface.co/settings/tokens)

Usage:
    from embedder import text_to_vector
    vec = text_to_vector("Hello, how are you?")   # returns list[float]
"""

import os
import requests

# ---------------------------------------------------------------------------
# HuggingFace Inference API Configuration
# ---------------------------------------------------------------------------
HF_API_URL = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction/"
    "sentence-transformers/all-MiniLM-L6-v2"
)
HF_TOKEN = os.getenv("HF_TOKEN")

print("[embedder] Using HuggingFace Inference API for embeddings")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def text_to_vector(text: str) -> list[float]:
    """
    Convert a string of text into a 384-dimensional vector embedding
    via the HuggingFace Inference API.

    Parameters
    ----------
    text : str
        The raw text to embed (a chat message, a query, etc.).

    Returns
    -------
    list[float]
        A list of 384 floating-point numbers representing the semantic
        meaning of the input text.

    Raises
    ------
    RuntimeError
        If the API call fails or returns an unexpected format.
    """
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text, "options": {"wait_for_model": True}}

    response = requests.post(HF_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise RuntimeError(
            f"HuggingFace API error {response.status_code}: {response.text}"
        )

    result = response.json()

    # The API returns a nested list for single inputs: [[0.1, 0.2, ...]]
    # We need the inner list.
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], list):
            return result[0]      # [[0.1, ...]] → [0.1, ...]
        return result             # [0.1, ...] already flat

    raise RuntimeError(f"Unexpected API response format: {result}")
