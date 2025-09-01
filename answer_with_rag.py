# answer_with_rag.py
from pathlib import Path
from typing import List, Tuple
import pickle

import faiss
import numpy as np
import openai
import streamlit as st

# Streamlit Secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Paths
EMBEDDING_PATH = Path("embeddings/faiss.index")
METADATA_PATH = Path("embeddings/metadata.pkl")

# Constants
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
# We'll try multiple thresholds in descending order (cosine/IP on normalized vectors)
ADAPTIVE_THRESHOLDS = [0.30, 0.20, 0.12, 0.08]

def _normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n == 0:
        return vec
    return vec / n

def _file_mtimes() -> Tuple[int, int]:
    """Return mtime in ns for index and metadata to key the cache."""
    idx_m = EMBEDDING_PATH.stat().st_mtime_ns if EMBEDDING_PATH.exists() else 0
    meta_m = METADATA_PATH.stat().st_mtime_ns if METADATA_PATH.exists() else 0
    return idx_m, meta_m

@st.cache_resource(show_spinner=False)
def load_index_and_meta(_idx_mtime: int, _meta_mtime: int):
    """Cache is invalidated automatically when the file mtimes change."""
    if not EMBEDDING_PATH.exists() or not METADATA_PATH.exists():
        return None, {}
    idx = faiss.read_index(str(EMBEDDING_PATH))
    with open(METADATA_PATH, "rb") as f:
        meta = pickle.load(f)
    return idx, meta

def get_embedding(text: str) -> np.ndarray:
    response = openai.Embedding.create(model=EMBED_MODEL, input=text)
    vec = np.asarray(response["data"][0]["embedding"], dtype=np.float32)
    return _normalize(vec)

def _search(index, query_vec: np.ndarray, k: int):
    # Ensure k doesn’t exceed index size
    ntotal = index.ntotal
    if ntotal == 0:
        # Empty index
        return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)
    eff_k = min(k, ntotal)
    return index.search(query_vec, eff_k)

def answer(question: str, k: int = 7, chat_history: List[dict] = []) -> str:
    idx_mtime, meta_mtime = _file_mtimes()
    index, metadata = load_index_and_meta(idx_mtime, meta_mtime)
    if index is None or not metadata:
        return "Embeddings not found. Go to 'Refresh Data' and rebuild the knowledge base."

    query_vec = get_embedding(question).reshape(1, -1)

    # Try adaptive thresholds
    D, I = _search(index, query_vec, k)
    chosen_items = []
    for threshold in ADAPTIVE_THRESHOLDS:
        tmp = []
        for i, score in zip(I[0], D[0]):
            if i == -1:
                continue
            if i in metadata and float(score) >= threshold:
                tmp.append((i, float(score)))
        if tmp:
            chosen_items = tmp
            break

    # If still empty after adaptive thresholds, we keep “Not found…” behavior
    if not chosen_items:
        return "Not found in documents."

    # Format context
    # Sort by score desc just in case FAISS returned unordered (usually it is ordered)
    chosen_items.sort(key=lambda x: x[1], reverse=True)
    context_text = "\n\n".join(metadata[i].get("text_preview", "") for i, _ in chosen_items)

    # Final prompt for GPT
    prompt = f"""
You are an executive assistant AI that answers strictly from the provided company documents.
Do not hallucinate or guess. If the answer isn't in the documents, reply exactly: "Not found in documents."

[Context from Documents]
{context_text}

[User Question]
{question}
""".strip()

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant for corporate knowledge based on internal documents."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return completion["choices"][0]["message"]["content"]
