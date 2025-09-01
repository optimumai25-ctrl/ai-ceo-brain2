import os
import pickle
from typing import List
from pathlib import Path

import faiss
import numpy as np
import openai
import streamlit as st

# Load OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Paths
EMBEDDING_PATH = Path("embeddings/faiss.index")
METADATA_PATH = Path("embeddings/metadata.pkl")

# Constants
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
SIMILARITY_THRESHOLD = 0.70  # Distance threshold for relevance filtering

# Load index + metadata
index = faiss.read_index(str(EMBEDDING_PATH))
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

def get_embedding(text: str) -> np.ndarray:
    response = openai.Embedding.create(
        model=EMBED_MODEL,
        input=text
    )
    vec = response["data"][0]["embedding"]
    return np.array(vec, dtype=np.float32)

def answer(question: str, k: int = 5, chat_history: List[dict] = []) -> str:
    query_vec = get_embedding(question)
    query_vec = np.array([query_vec], dtype=np.float32)

    # Top-k similarity search
    D, I = index.search(query_vec, k)
    top_k = [
        metadata[i] for i, dist in zip(I[0], D[0])
        if i in metadata and dist < SIMILARITY_THRESHOLD
    ]

    # Format context from matching documents
    context_text = "\n\n".join(
        f"{item['text_preview']}"
        for item in top_k
    )

    # Include previous chat if needed
    history_text = "\n".join(
        f"{h['role'].capitalize()}: {h['content']}"
        for h in chat_history if h['role'] in ("user", "assistant")
    )

    # Final prompt for GPT
    prompt = f"""
You are an executive assistant AI that answers based on company documents.
Do not hallucinate or guess. If the answer isn't in the documents, say: "Not found in documents."

[Context from Documents]
{context_text}

[User Question]
{question}
""".strip()

    # Call GPT
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant for corporate knowledge based on internal documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return completion["choices"][0]["message"]["content"]

