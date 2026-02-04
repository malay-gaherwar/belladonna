import os
from dataclasses import dataclass
from typing import Any, List, Tuple

import streamlit as st
from openai import OpenAI

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Config
# -----------------------------
TEXT_PATH = "/home/malay/Documents/Belladonna/message.txt"

BASE_URL = os.environ.get("BASE_URL", "http://192.168.33.27/v1").strip()
API_KEY = os.environ.get("VIRTUAL_API_KEY", None)

MODEL = "GPT-OSS-120B"
MAX_COMPLETION_TOKENS = 1024

TOP_K_DEFAULT = 5           # number of chunks to retrieve
CHUNK_SIZE = 1200           # characters
CHUNK_OVERLAP = 200         # characters


# -----------------------------
# Utilities
# -----------------------------
def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        return [text]
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


@dataclass
class RAGIndex:
    chunks: List[str]
    vectorizer: TfidfVectorizer
    matrix: Any  # <-- FIX: dataclass fields must have type annotations


def build_index(chunks: List[str]) -> RAGIndex:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(chunks)
    return RAGIndex(chunks=chunks, vectorizer=vectorizer, matrix=matrix)


def retrieve(index: RAGIndex, query: str, top_k: int) -> List[Tuple[int, float, str]]:
    qv = index.vectorizer.transform([query])
    sims = cosine_similarity(qv, index.matrix).ravel()
    top_idx = sims.argsort()[::-1][:top_k]
    return [(int(i), float(sims[i]), index.chunks[int(i)]) for i in top_idx]


def format_context(snips: List[Tuple[int, float, str]]) -> str:
    parts = []
    for idx, score, chunk in snips:
        parts.append(f"[CHUNK {idx} | score={score:.3f}]\n{chunk}")
    return "\n\n---\n\n".join(parts)


def make_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


SYSTEM_PROMPT = """You are a clinical RAG assistant.

CRITICAL RULES:
1) You MUST answer ONLY using the provided CONTEXT (from a single text file).
2) If the answer is not explicitly supported by the CONTEXT, you must say:
   "I don't know based on the provided file."
3) Do NOT use outside knowledge. Do NOT guess. Do NOT add medical advice beyond the text.
4) Always include citations to the context by referencing chunk IDs like [CHUNK 12].
5) Keep answers concise and clinician-friendly.
"""

USER_PROMPT_TEMPLATE = """QUESTION:
{question}

CONTEXT (the ONLY source you may use):
{context}

INSTRUCTIONS:
- Answer only from CONTEXT.
- If not in CONTEXT: say "I don't know based on the provided file."
- Provide citations like [CHUNK X] for each key statement.
"""


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Belladonna RAG (File-only)", layout="wide")
st.title("Belladonna RAG (answers only from message.txt)")

with st.sidebar:
    st.header("Settings")
    st.write(f"**File:** `{TEXT_PATH}`")
    st.write(f"**BASE_URL:** `{BASE_URL}`")
    st.write(f"**Model:** `{MODEL}`")

    top_k = st.slider("Top-K chunks", 1, 10, TOP_K_DEFAULT, 1)
    max_tokens = st.slider("Max completion tokens", 128, 2048, MAX_COMPLETION_TOKENS, 64)

    show_retrieval = st.checkbox("Show retrieved chunks", value=True)
    rebuild = st.button("Rebuild index")


@st.cache_resource(show_spinner=True)
def load_and_index(path: str) -> RAGIndex:
    text = clean_text(read_text(path))
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    return build_index(chunks)


if rebuild:
    load_and_index.clear()

try:
    index = load_and_index(TEXT_PATH)
except Exception as e:
    st.error(f"Failed to load/index file: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask a question about the content of message.txt…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    snips = retrieve(index, question, top_k=top_k)
    context = format_context(snips)

    if show_retrieval:
        with st.expander("Retrieved context (what the model is allowed to use)"):
            for idx, score, chunk in snips:
                st.markdown(f"**CHUNK {idx}** (score={score:.3f})")
                st.code(chunk[:4000])

    with st.chat_message("assistant"):
        try:
            client = make_client()
            prompt = USER_PROMPT_TEMPLATE.format(question=question, context=context)

            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=MODEL,
                max_completion_tokens=max_tokens,
            )

            answer = response.choices[0].message.content.strip()

            # Extra guardrail: require citations unless it's an explicit "don't know".
            if ("CHUNK" not in answer) and ("don't know based on the provided file" not in answer.lower()):
                answer = "I don't know based on the provided file."

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"LLM call failed: {e}")
            fallback = "I don't know based on the provided file."
            st.session_state.messages.append({"role": "assistant", "content": fallback})
