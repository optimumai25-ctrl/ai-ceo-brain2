# answer_with_rag.py
from pathlib import Path
from typing import List

import faiss
import numpy as np
import openai
import streamlit as st
import pickle

# Streamlit Secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Paths
EMBEDDING_PATH = Path("embeddings/faiss.index")
METADATA_PATH = Path("embeddings/metadata.pkl")

# Constants
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
# Cosine similarity threshold (since we use IndexFlatIP with normalized vectors)
SIMILARITY_THRESHOLD = 0.30

def _normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n == 0:
        return vec
    return vec / n

@st.cache_resource
def load_index_and_meta():
    if not EMBEDDING_PATH.exists() or not METADATA_PATH.exists():
        return None, {}
    idx = faiss.read_index(str(EMBEDDING_PATH))
    with open(METADATA_PATH, "rb") as f:
        meta = pickle.load(f)
    return idx, meta

def get_embedding(text: str) -> np.ndarray:
    response = openai.Embedding.create(model=EMBED_MODEL, input=text)
    vec = np.asarray(response["data"][0]["embedding"], dtype=np.float32)
    vec = _normalize(vec)
    return vec

def answer(question: str, k: int = 5, chat_history: List[dict] = []) -> str:
    index, metadata = load_index_and_meta()
    if index is None or not metadata:
        return "Embeddings not found. Go to 'Refresh Data' and rebuild the knowledge base."

    query_vec = get_embedding(question).reshape(1, -1)

    # Top-k similarity search
    D, I = index.search(query_vec, k)
    # Since we use IP on normalized vectors, D are cosine similarities in [-1, 1]
    top_k = []
    for i, score in zip(I[0], D[0]):
        if i == -1:
            continue
        if i in metadata and float(score) >= SIMILARITY_THRESHOLD:
            top_k.append(metadata[i])

    if not top_k:
        context_text = ""
    else:
        context_text = "\n\n".join(item.get("text_preview", "") for item in top_k)

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

