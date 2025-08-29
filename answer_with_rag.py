import os
import pickle
from typing import List
from pathlib import Path

import faiss
import numpy as np
import openai
from dotenv import load_dotenv

# Load .env if needed
load_dotenv()

# Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Paths
EMBEDDING_PATH = Path("embeddings/faiss.index")
METADATA_PATH = Path("embeddings/metadata.pkl")

# Constants
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

def load_index_and_metadata():
    if not EMBEDDING_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("FAISS index or metadata not found. Please refresh data first.")

    index = faiss.read_index(str(EMBEDDING_PATH))
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata

def get_embedding(text: str) -> np.ndarray:
    response = openai.Embedding.create(
        model=EMBED_MODEL,
        input=text
    )
    vec = response["data"][0]["embedding"]
    return np.array(vec, dtype=np.float32)

def answer(question: str, k: int = 5, chat_history: List[dict] = []) -> str:
    # Load FAISS index and metadata on demand
    index, metadata = load_index_and_metadata()

    query_vec = get_embedding(question)
    query_vec = np.array([query_vec], dtype=np.float32)

    # Top-k similarity search
    D, I = index.search(query_vec, k)
    top_k = [metadata[i] for i in I[0] if i in metadata]

    # Format context (remove source tags for clean prompts)
    context_text = "\n\n".join(
        f"{item['text_preview']}"
        for item in top_k
    )

    # Optional: Chat history formatting
    history_text = "\n".join(
        f"{h['role'].capitalize()}: {h['content']}"
        for h in chat_history if h['role'] in ("user", "assistant")
    )

    prompt = f"""
You are an executive assistant AI that answers based strictly on company documents.
Do not guess or hallucinate. If the answer isn't in the documents, say \"Not found in documents.\"

[Context from Documents]
{context_text}

[Question]
{question}
""".strip()

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant for corporate knowledge based on internal documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return completion["choices"][0]["message"]["content"]

