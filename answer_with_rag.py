# answer_with_rag.py
from pathlib import Path
from typing import List, Tuple
import pickle
import re

import faiss
import numpy as np
import openai
import streamlit as st

# Streamlit Secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Paths
EMBEDDING_PATH = Path("embeddings/faiss.index")
METADATA_PATH = Path("embeddings/metadata.pkl")
PARSED_DIR = Path("parsed_data")

# Constants
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
# Adaptive cosine/IP thresholds on normalized vectors
ADAPTIVE_THRESHOLDS = [0.30, 0.22, 0.15, 0.10, 0.06]

# -------- Utils --------
def _normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec if n == 0 else (vec / n)

def _file_mtimes() -> Tuple[int, int]:
    idx_m = EMBEDDING_PATH.stat().st_mtime_ns if EMBEDDING_PATH.exists() else 0
    meta_m = METADATA_PATH.stat().st_mtime_ns if METADATA_PATH.exists() else 0
    return idx_m, meta_m

@st.cache_resource(show_spinner=False)
def _load_index_and_meta(_idx_mtime: int, _meta_mtime: int):
    if not EMBEDDING_PATH.exists() or not METADATA_PATH.exists():
        return None, {}
    idx = faiss.read_index(str(EMBEDDING_PATH))
    with open(METADATA_PATH, "rb") as f:
        meta = pickle.load(f)
    return idx, meta

def _get_embedding(text: str) -> np.ndarray:
    r = openai.Embedding.create(model=EMBED_MODEL, input=text)
    v = np.asarray(r["data"][0]["embedding"], dtype=np.float32)
    return _normalize(v)

def _search(index, query_vec: np.ndarray, k: int):
    ntotal = index.ntotal
    if ntotal == 0:
        return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)
    eff_k = min(k, ntotal)
    return index.search(query_vec, eff_k)

def _split_paragraphs(txt: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", txt) if p.strip()]
    return paras

def _keyword_fuzzy_score(paragraph: str, tokens: List[str]) -> float:
    # Very light heuristic score: term hits + partials
    p = paragraph.lower()
    score = 0.0
    for t in tokens:
        if not t:
            continue
        tl = t.lower()
        if tl in p:
            score += 1.0
        else:
            # partial match heuristic
            if len(tl) >= 4:
                if any(tl[:m] in p for m in (4, 5)):
                    score += 0.4
    return score

def _keyword_sweep(question: str, max_paras: int = 12) -> List[str]:
    """
    Grep-style pass over parsed_data to find best paragraphs when vectors miss.
    Returns top paragraphs concatenated later.
    """
    if not PARSED_DIR.exists():
        return []

    # Basic tokenization: words >= 3 chars
    tokens = [w for w in re.findall(r"[A-Za-z0-9_]+", question) if len(w) >= 3]

    hits = []
    for txt_path in PARSED_DIR.rglob("*.txt"):
        try:
            txt = txt_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for para in _split_paragraphs(txt):
            s = _keyword_fuzzy_score(para, tokens)
            if s > 0:
                hits.append((s, para))

    if not hits:
        return []

    # Sort by score descending, keep top N distinct paragraphs
    hits.sort(key=lambda x: x[0], reverse=True)
    dedup = []
    seen = set()
    for s, para in hits:
        key = para[:160]  # crude dedup by prefix
        if key in seen:
            continue
        seen.add(key)
        dedup.append(para)
        if len(dedup) >= max_paras:
            break
    return dedup

# -------- Public API --------
def answer(question: str, k: int = 7, chat_history: List[dict] = [], strict_mode: bool = False) -> str:
    """
    strict_mode=False => enable keyword sweep + general-knowledge fallback.
    strict_mode=True  => vector-only; return 'Not found in documents.' if no hits.
    """
    # 1) Vector search (adaptive thresholds)
    idx_mtime, meta_mtime = _file_mtimes()
    index, metadata = _load_index_and_meta(idx_mtime, meta_mtime)

    vector_context_chunks: List[str] = []
    if index is not None and metadata:
        qv = _get_embedding(question).reshape(1, -1)
        D, I = _search(index, qv, k)

        chosen = []
        for thr in ADAPTIVE_THRESHOLDS:
            tmp = []
            for i, score in zip(I[0], D[0]):
                if i == -1:
                    continue
                if i in metadata and float(score) >= thr:
                    tmp.append((i, float(score)))
            if tmp:
                chosen = tmp
                break

        if chosen:
            chosen.sort(key=lambda x: x[1], reverse=True)
            vector_context_chunks = [metadata[i].get("text_preview", "") for i, _ in chosen]

    # 2) If no vector hits and not strict, do keyword/fuzzy sweep
    keyword_context_chunks: List[str] = []
    if not vector_context_chunks and not strict_mode:
        keyword_context_chunks = _keyword_sweep(question, max_paras=12)

    # 3) Build prompt paths
    if vector_context_chunks or keyword_context_chunks:
        context_text = "\n\n".join(vector_context_chunks + keyword_context_chunks)
        prompt = f"""
You are an executive assistant AI that answers strictly from the provided company documents below.
If the answer is not supported by the context, reply exactly: "Not found in documents."

[Context from Documents]
{context_text}

[User Question]
{question}
""".strip()
    else:
        if strict_mode:
            return "Not found in documents."
        # General-knowledge fallback with explicit assumptions
        prompt = f"""
You are an executive assistant AI. The user asked a question but there were no matching internal documents.
Provide a practical, structured answer using standard business knowledge and clearly label all assumptions.
Your output must include:
1) A concise meeting summary
2) Financial highlights and risks
3) Action items with owners and due dates
4) Open questions for follow-up
Add a final note: "No matching internal documents were found; the above includes assumptions."

[User Question]
{question}
""".strip()

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant for corporate knowledge. Prefer internal documents when available; otherwise provide a clearly labeled, assumption-based answer."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return completion["choices"][0]["message"]["content"]
