"""Chat Router — with CRAG verdict, filtered retrieval, and query routing."""
import time, structlog
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from services.rag_service import RAGService
from services.metrics import (
    rag_requests_total, rag_request_duration_seconds,
    rag_crag_verdicts_total, rag_query_routing_total,
)
from dependencies import get_rag_service

logger = structlog.get_logger()
router = APIRouter(tags=["Chat"])

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    content_type_filter: str | None = Field(None, description="Filter by: methodology, results, defense, attack, background, practical")

class SourceRef(BaseModel):
    source: str = ""
    page: int | None = None
    content_type: str = "unknown"
    topic: str = ""
    audience: str = ""
    snippet: str = ""

class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceRef]
    verdict: str
    refined_query: str | None = None
    attempts: int
    routed_audience: str | None = None

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, svc: RAGService = Depends(get_rag_service)):
    t0 = time.time()
    try:
        r = svc.query(request.query, content_type_filter=request.content_type_filter)

        # Record metrics
        verdict = r.get("verdict", "CORRECT")
        audience = r.get("routed_audience")

        rag_crag_verdicts_total.labels(verdict=verdict).inc()
        rag_query_routing_total.labels(audience=audience or "none").inc()
        rag_requests_total.labels(endpoint="/chat", status="200").inc()
        rag_request_duration_seconds.labels(endpoint="/chat").observe(time.time() - t0)

        return ChatResponse(
            query=request.query, answer=r["answer"],
            sources=[SourceRef(**s) for s in r.get("sources", [])],
            verdict=verdict,
            refined_query=r.get("refined_query"), attempts=r.get("attempts", 1),
            routed_audience=audience,
        )
    except RuntimeError:
        rag_requests_total.labels(endpoint="/chat", status="503").inc()
        raise HTTPException(503, "Pipeline not initialized. Ingest a document first.")
    except Exception as e:
        rag_requests_total.labels(endpoint="/chat", status="500").inc()
        logger.error("chat_failed", error=str(e), exc_info=True)
        raise HTTPException(500, str(e))