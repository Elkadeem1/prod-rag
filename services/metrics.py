"""Prometheus Metrics for RAG pipeline observability."""
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ── Counters ─────────────────────────────────────────────────────────────────
rag_requests_total = Counter(
    "rag_requests_total", "Total API requests", ["endpoint", "status"]
)
rag_chunks_created_total = Counter(
    "rag_chunks_created_total", "Chunks created by strategy", ["strategy"]
)
rag_crag_verdicts_total = Counter(
    "rag_crag_verdicts_total", "CRAG verdict distribution", ["verdict"]
)
rag_query_routing_total = Counter(
    "rag_query_routing_total", "Query routing decisions", ["audience"]
)
rag_enrichment_total = Counter(
    "rag_enrichment_total", "Enrichment operations", ["mode"]
)

# ── Histograms ───────────────────────────────────────────────────────────────
rag_request_duration_seconds = Histogram(
    "rag_request_duration_seconds", "Request latency", ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
rag_enrichment_duration_seconds = Histogram(
    "rag_enrichment_duration_seconds", "Enrichment latency", ["mode"],
    buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.5, 5.0],
)
rag_chunking_duration_seconds = Histogram(
    "rag_chunking_duration_seconds", "Chunking latency", ["strategy"],
    buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)


def get_metrics() -> bytes:
    """Return Prometheus metrics in text format."""
    return generate_latest()


def get_content_type() -> str:
    return CONTENT_TYPE_LATEST