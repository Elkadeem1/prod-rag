"""Debug Router — Collection stats and query routing debugger."""
import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from services.rag_service import detect_query_audience, AUDIENCE_KEYWORDS, AUDIENCE_TO_DOCTYPE
from dependencies import get_vector_store
from config import get_settings

logger = structlog.get_logger()
router = APIRouter(tags=["Debug"])


# ── Collection Stats ─────────────────────────────────────────────────────────

class CollectionStats(BaseModel):
    collection: str
    total_chunks: int
    metadata_fields: list[str] = Field(default_factory=list)
    content_type_distribution: dict[str, int] = Field(default_factory=dict)
    audience_distribution: dict[str, int] = Field(default_factory=dict)
    doc_type_distribution: dict[str, int] = Field(default_factory=dict)
    has_numbers_count: int = 0
    has_table_count: int = 0
    has_code_count: int = 0
    avg_word_count: float = 0

@router.get("/stats", response_model=CollectionStats)
async def collection_stats():
    """Get collection statistics showing metadata coverage and distributions."""
    vs = get_vector_store()
    settings = get_settings()
    col_name = settings.qdrant.collection_name

    try:
        info = vs._client.get_collection(col_name)
        total = info.points_count or 0
    except Exception:
        return CollectionStats(collection=col_name, total_chunks=0)

    if total == 0:
        return CollectionStats(collection=col_name, total_chunks=0)

    # Sample points to compute distributions
    try:
        points, _ = vs._client.scroll(
            col_name, limit=min(total, 500), with_payload=True, with_vectors=False,
        )
    except Exception:
        return CollectionStats(collection=col_name, total_chunks=total)

    content_type_dist = {}
    audience_dist = {}
    doc_type_dist = {}
    has_numbers = 0
    has_table = 0
    has_code = 0
    word_counts = []
    all_fields = set()

    for p in points:
        payload = p.payload or {}
        meta = payload.get("metadata", payload)
        all_fields.update(meta.keys())

        ct = meta.get("content_type", "unknown")
        content_type_dist[ct] = content_type_dist.get(ct, 0) + 1

        aud = meta.get("audience", "unknown")
        audience_dist[aud] = audience_dist.get(aud, 0) + 1

        dt = meta.get("doc_type", "unknown")
        doc_type_dist[dt] = doc_type_dist.get(dt, 0) + 1

        if meta.get("has_numbers"):
            has_numbers += 1
        if meta.get("has_table"):
            has_table += 1
        if meta.get("has_code"):
            has_code += 1
        wc = meta.get("word_count", 0)
        if wc:
            word_counts.append(wc)

    return CollectionStats(
        collection=col_name,
        total_chunks=total,
        metadata_fields=sorted(all_fields),
        content_type_distribution=dict(sorted(content_type_dist.items(), key=lambda x: -x[1])),
        audience_distribution=dict(sorted(audience_dist.items(), key=lambda x: -x[1])),
        doc_type_distribution=dict(sorted(doc_type_dist.items(), key=lambda x: -x[1])),
        has_numbers_count=has_numbers,
        has_table_count=has_table,
        has_code_count=has_code,
        avg_word_count=round(sum(word_counts) / max(len(word_counts), 1), 1),
    )


# ── Query Routing Debugger ───────────────────────────────────────────────────

class RoutingDebugRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)

class RoutingDebugResponse(BaseModel):
    query: str
    detected_audience: str | None
    routed_to: str | None
    keyword_matches: list[str] = Field(default_factory=list)
    all_scores: dict[str, int] = Field(default_factory=dict)
    explanation: str = ""

@router.post("/debug/routing", response_model=RoutingDebugResponse)
async def debug_routing(req: RoutingDebugRequest):
    """Show exactly how query routing works — which keywords match, which audience wins."""
    q = req.query.lower()
    audience = detect_query_audience(req.query)

    # Compute detailed scores
    all_scores = {}
    keyword_matches = []
    for aud, keywords in AUDIENCE_KEYWORDS.items():
        matches = [k for k in keywords if k in q]
        all_scores[aud] = len(matches)
        if matches:
            keyword_matches.extend([f"{aud}:{m}" for m in matches])

    routed_to = AUDIENCE_TO_DOCTYPE.get(audience) if audience else None

    # Build explanation
    if audience:
        explanation = (
            f"Query matched {all_scores[audience]} '{audience}' keyword(s): "
            f"{[k for k in AUDIENCE_KEYWORDS[audience] if k in q]}. "
            f"Routing to '{routed_to}' collection filter."
        )
    else:
        explanation = (
            "No audience keywords matched. Query will search ALL documents (no routing). "
            "This is typical for cross-domain queries."
        )

    return RoutingDebugResponse(
        query=req.query,
        detected_audience=audience,
        routed_to=routed_to,
        keyword_matches=keyword_matches,
        all_scores=all_scores,
        explanation=explanation,
    )