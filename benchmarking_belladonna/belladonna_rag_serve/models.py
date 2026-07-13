from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    sources: Optional[List[str]] = None
    top_k: int = 15
    # Optional per-request override of the answer-generation LLM. When None,
    # the server's configured MODEL_NAME is used. Lets a benchmark compare the
    # RAG across backing LLMs without restarting the server.
    model: Optional[str] = None


class EvidenceItem(BaseModel):
    source: str
    file_name: str
    factoid_text: str
    score: float
    rerank_score: float = 0.0
    display_title: str = ""
    doi: str = ""
    document_year: str = ""
    source_family: str = ""
    document_type: str = ""
    source_pdf_name: str = ""
    citation_label: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceGap(BaseModel):
    aspect: str
    follow_up_query: str


class EvidenceVerdict(BaseModel):
    sufficient: bool
    confidence: float = 0.0
    gaps: List[EvidenceGap] = Field(default_factory=list)
    notes: str = ""
    source: str = ""   # "llm" | "empty" | "error" | "parse-error"


class QueryResponse(BaseModel):
    question: str
    answer: str
    evidence: List[EvidenceItem]
    evidence_verdict: Optional[EvidenceVerdict] = None
    routed_sources: List[str] = Field(default_factory=list)
    routing_method: str = ""
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    sources: Optional[List[str]] = None
    top_k: int = 15


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    # The history-resolved standalone query actually used for retrieval.
    search_query: str
    evidence: List[EvidenceItem]
    history: List[ChatMessage]
    evidence_verdict: Optional[EvidenceVerdict] = None
    routed_sources: List[str] = Field(default_factory=list)
    routing_method: str = ""


class ResetRequest(BaseModel):
    session_id: str


class ResetResponse(BaseModel):
    session_id: str
    status: str