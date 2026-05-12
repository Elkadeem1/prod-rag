"""Compare Router — Side-by-side comparison of chunking, enrichment, and retrieval strategies."""
import time, structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from langchain.schema import Document
from services.chunking import FixedChunker, RecursiveChunker, MarkdownAwareChunker, SemanticChunker
from services.metadata import ProductionEnricher, MetadataEnricher
from services.rag_service import RAGService, detect_query_audience, AUDIENCE_TO_DOCTYPE
from services.metrics import rag_chunking_duration_seconds, rag_enrichment_duration_seconds
from dependencies import get_rag_service, get_vector_store
from config import get_settings

logger = structlog.get_logger()
router = APIRouter(tags=["Comparison"])

# ── Chunking Comparison ──────────────────────────────────────────────────────

class ChunkingCompareRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=50000, description="Text to chunk")
    chunk_size: int = Field(default=500, gt=50, le=2000)
    chunk_overlap: int = Field(default=50, ge=0)
    format: str = Field(default="text", description="text or markdown — affects markdown-aware strategy")

class StrategyResult(BaseModel):
    chunk_count: int
    avg_chars: float
    min_chars: int
    max_chars: int
    latency_ms: float
    samples: list[str] = Field(default_factory=list, description="First 3 chunks (truncated)")

class ChunkingCompareResponse(BaseModel):
    strategies: dict[str, StrategyResult]
    recommendation: str

@router.post("/compare/chunking", response_model=ChunkingCompareResponse)
async def compare_chunking(req: ChunkingCompareRequest):
    """Run the same text through all chunking strategies and compare results."""
    doc = Document(
        page_content=req.text,
        metadata={"page": 1, "source": "comparison_input", "format": req.format},
    )

    strategies = {
        "fixed": FixedChunker(chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap),
        "recursive": RecursiveChunker(chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap),
        "markdown": MarkdownAwareChunker(chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap),
    }

    results = {}
    for name, chunker in strategies.items():
        t0 = time.time()
        chunks = chunker.chunk([doc])
        latency = (time.time() - t0) * 1000
        rag_chunking_duration_seconds.labels(strategy=name).observe(latency / 1000)

        if chunks:
            char_counts = [len(c.page_content) for c in chunks]
            results[name] = StrategyResult(
                chunk_count=len(chunks),
                avg_chars=round(sum(char_counts) / len(char_counts), 1),
                min_chars=min(char_counts),
                max_chars=max(char_counts),
                latency_ms=round(latency, 1),
                samples=[c.page_content[:200] + "..." if len(c.page_content) > 200 else c.page_content for c in chunks[:3]],
            )
        else:
            results[name] = StrategyResult(chunk_count=0, avg_chars=0, min_chars=0, max_chars=0, latency_ms=round(latency, 1))

    # Recommendation logic
    best = min(results.items(), key=lambda x: abs(x[1].avg_chars - req.chunk_size) if x[1].chunk_count > 0 else 9999)
    recommendation = f"{best[0]} ({best[1].chunk_count} chunks, avg {best[1].avg_chars:.0f} chars)"
    if req.format == "markdown" and results.get("markdown", StrategyResult(chunk_count=0, avg_chars=0, min_chars=0, max_chars=0, latency_ms=0)).chunk_count > 0:
        recommendation = f"markdown ({results['markdown'].chunk_count} chunks, header-aware boundaries)"

    return ChunkingCompareResponse(strategies=results, recommendation=recommendation)


# ── Enrichment Comparison ────────────────────────────────────────────────────

class EnrichmentCompareRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=10000, description="Text to enrich")

class EnrichmentResult(BaseModel):
    content_type: str = ""
    audience: str = ""
    complexity: str = ""
    key_entities: list[str] = Field(default_factory=list)
    has_numbers: bool = False
    has_table: bool = False
    has_code: bool = False
    word_count: int = 0
    latency_ms: float = 0
    cost: str = "$0"

class EnrichmentCompareResponse(BaseModel):
    regex: EnrichmentResult
    llm: EnrichmentResult | None = None
    agreement: dict[str, bool] = Field(default_factory=dict)
    llm_skipped: bool = False
    llm_skip_reason: str = ""

@router.post("/compare/enrichment", response_model=EnrichmentCompareResponse)
async def compare_enrichment(req: EnrichmentCompareRequest):
    """Compare regex (free) vs LLM enrichment on the same text."""
    doc = Document(page_content=req.text, metadata={})

    # Regex enrichment
    prod = ProductionEnricher()
    t0 = time.time()
    prod.enrich(doc)
    regex_latency = (time.time() - t0) * 1000
    rag_enrichment_duration_seconds.labels(mode="regex").observe(regex_latency / 1000)

    regex_result = EnrichmentResult(
        content_type=doc.metadata.get("content_type", ""),
        audience=doc.metadata.get("audience", ""),
        complexity=doc.metadata.get("complexity", ""),
        key_entities=doc.metadata.get("key_entities", []),
        has_numbers=doc.metadata.get("has_numbers", False),
        has_table=doc.metadata.get("has_table", False),
        has_code=doc.metadata.get("has_code", False),
        word_count=doc.metadata.get("word_count", 0),
        latency_ms=round(regex_latency, 2),
        cost="$0",
    )

    # LLM enrichment (optional — requires API key)
    settings = get_settings()
    llm_result = None
    llm_skipped = False
    llm_skip_reason = ""

    try:
        llm_doc = Document(page_content=req.text, metadata={})
        enricher = MetadataEnricher(google_api_key=settings.gemini.api_key)
        t0 = time.time()
        enricher.enrich(llm_doc)
        llm_latency = (time.time() - t0) * 1000
        rag_enrichment_duration_seconds.labels(mode="llm").observe(llm_latency / 1000)

        llm_result = EnrichmentResult(
            content_type=llm_doc.metadata.get("content_type", llm_doc.metadata.get("topic", "")),
            audience=llm_doc.metadata.get("audience", ""),
            complexity=llm_doc.metadata.get("complexity", ""),
            key_entities=llm_doc.metadata.get("key_entities", []),
            has_numbers=llm_doc.metadata.get("has_data", False),
            latency_ms=round(llm_latency, 1),
            cost=f"~$0.004",
        )
    except Exception as e:
        llm_skipped = True
        llm_skip_reason = str(e)

    # Agreement
    agreement = {}
    if llm_result:
        agreement["content_type"] = regex_result.content_type == llm_result.content_type
        agreement["audience"] = regex_result.audience == llm_result.audience

    return EnrichmentCompareResponse(
        regex=regex_result, llm=llm_result,
        agreement=agreement, llm_skipped=llm_skipped, llm_skip_reason=llm_skip_reason,
    )


# ── Retrieval Comparison ─────────────────────────────────────────────────────

class RetrievalCompareRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)

class RetrievalPipelineResult(BaseModel):
    sources: list[dict] = Field(default_factory=list)
    latency_ms: float = 0
    audience: str | None = None
    routed_to: str | None = None
    verdict: str | None = None
    corrected: bool = False

class RetrievalCompareResponse(BaseModel):
    query: str
    naive: RetrievalPipelineResult
    routed: RetrievalPipelineResult
    crag: RetrievalPipelineResult
    insight: str

@router.post("/compare/retrieval", response_model=RetrievalCompareResponse)
async def compare_retrieval(req: RetrievalCompareRequest, svc: RAGService = Depends(get_rag_service)):
    """Compare naive, routed, and CRAG retrieval on the same query."""
    vs = get_vector_store()
    settings = get_settings()
    k = settings.retriever.initial_k

    # Pipeline 1: Naive (unfiltered search)
    t0 = time.time()
    naive_docs = vs.search(req.query, k=k)
    naive_latency = (time.time() - t0) * 1000
    naive = RetrievalPipelineResult(
        sources=[{"source": d.metadata.get("source", "?"), "page": d.metadata.get("page"), "snippet": d.page_content[:100]} for d in naive_docs[:5]],
        latency_ms=round(naive_latency, 1),
    )

    # Pipeline 2: Routed
    t0 = time.time()
    audience = detect_query_audience(req.query)
    if audience and audience in AUDIENCE_TO_DOCTYPE:
        target = AUDIENCE_TO_DOCTYPE[audience]
        routed_docs = vs.filtered_search(req.query, "doc_type", target, k=k)
        if not routed_docs:
            routed_docs = vs.search(req.query, k=k)
    else:
        routed_docs = vs.search(req.query, k=k)
        target = None
    routed_latency = (time.time() - t0) * 1000
    routed = RetrievalPipelineResult(
        sources=[{"source": d.metadata.get("source", "?"), "page": d.metadata.get("page"), "snippet": d.page_content[:100]} for d in routed_docs[:5]],
        latency_ms=round(routed_latency, 1),
        audience=audience, routed_to=target,
    )

    # Pipeline 3: Full CRAG
    t0 = time.time()
    crag_result = svc.query(req.query)
    crag_latency = (time.time() - t0) * 1000
    crag = RetrievalPipelineResult(
        sources=crag_result.get("sources", [])[:5],
        latency_ms=round(crag_latency, 1),
        verdict=crag_result.get("verdict"),
        corrected=crag_result.get("attempts", 1) > 1,
        audience=crag_result.get("routed_audience"),
    )

    # Insight
    speedup = round(crag.latency_ms / max(naive.latency_ms, 1), 1)
    insight = f"CRAG is {speedup}x slower than naive ({crag.latency_ms:.0f}ms vs {naive.latency_ms:.0f}ms). "
    if audience:
        insight += f"Query routed to '{target}' based on '{audience}' audience. "
    if crag.corrected:
        insight += "CRAG triggered self-correction. "

    return RetrievalCompareResponse(
        query=req.query, naive=naive, routed=routed, crag=crag, insight=insight,
    )