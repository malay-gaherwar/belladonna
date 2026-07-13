from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from models import (
    QueryRequest,
    QueryResponse,
    ChatRequest,
    ChatResponse,
    ResetRequest,
    ResetResponse,
)
from answerer import answer_question, chat_answer
from conversation import STORE
from retriever import source_status
from source_info import get_source_info
from config import SOURCE_FACTOID_DIRS, MODEL_NAME

app = FastAPI(title="Belladonna RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Frontend (the chat UI lives at ../rag/index.html, with shared assets under
# ../images and ../ for the landing page). Mounting the whole site directory
# at /site lets every relative href in index.html ("../images/...", "../")
# resolve correctly. Visiting / redirects to the chat UI so one URL is all
# the user has to remember.
SITE_DIR = (Path(__file__).resolve().parent.parent).resolve()


@app.get("/")
def root():
    return RedirectResponse(url="/site/rag/")


@app.get("/api")
def api_info():
    return {
        "message": "Belladonna RAG API running",
        "model": MODEL_NAME,
        "sources": list(SOURCE_FACTOID_DIRS.keys()),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sources")
def sources():
    return {"sources": list(SOURCE_FACTOID_DIRS.keys())}


@app.get("/status")
def status():
    """Per-source vector-store health (which collections actually load)."""
    return {"sources": source_status()}


@app.get("/source_info")
def source_info():
    """Per-source overview for the website sidebar (cached on first call):
    health, document counts, years (guidelines/regulators), article counts
    (papers), trial counts (CTG)."""
    return {"sources": get_source_info()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer_question(
            question=req.question,
            sources=req.sources,
            top_k=req.top_k,
            model=req.model,
        )
    except HTTPException:
        raise
    except Exception as exc:  # keep the server alive on transient upstream errors
        raise HTTPException(status_code=503, detail=(f"upstream error: {type(exc).__name__}: {exc}")[:300])
    return result


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Multi-turn chat. Pass back the returned session_id on the next
    message to keep conversation memory; omit it to start fresh."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    return chat_answer(
        message=req.message,
        session_id=req.session_id,
        sources=req.sources,
        top_k=req.top_k,
    )


@app.post("/chat/reset", response_model=ResetResponse)
def chat_reset(req: ResetRequest):
    """Forget a conversation's memory."""
    STORE.reset(req.session_id)
    return {"session_id": req.session_id, "status": "reset"}


# IMPORTANT: this mount must stay last. StaticFiles at "/site" only ever
# matches paths under that prefix, so it does not shadow the API routes
# above, but registering it after them keeps the precedence obvious.
app.mount("/site", StaticFiles(directory=SITE_DIR, html=True), name="site")