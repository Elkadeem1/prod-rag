"""RAG Service — Gemini LLM with metadata enrichment, reranking, query routing, and CRAG."""
import re, structlog
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from config import AppSettings
from database.vector_store import VectorStoreConnector
from services.chunking import ChunkingStrategy
from services.metadata import ProductionEnricher, MetadataEnricher
from services.crag import CRAGEvaluator

logger = structlog.get_logger()

GROUNDED_PROMPT = """Answer using ONLY the context below. Cite [Source: filename] for every claim.
If the context doesn't contain the answer, say "I cannot answer from the provided context."

Context:
{context}

Question: {question}
Answer:"""

RERANK_PROMPT = "Rate relevance 0-10 (integer only).\nQuery: {query}\nChunk: {chunk}\nScore:"

# ── Query Audience Routing (from NB02 Section 4) ────────────────────────────

AUDIENCE_KEYWORDS = {
    'technical': ['deploy', 'tier', 'latency', 'cost', 'production', 'architecture', 'config'],
    'academic': ['et al', 'study', 'experiment', 'paper', 'research', 'hypothesis', 'table'],
    'general': ['how do i', 'can i', 'what is', 'help', 'fix', 'stop', 'simple', 'my'],
}

AUDIENCE_TO_DOCTYPE = {
    'technical': 'internal_wiki',
    'academic': 'academic_paper',
    'general': 'customer_faq',
}


def detect_query_audience(query: str) -> str | None:
    """Classify query intent to route to the right document type."""
    q = query.lower()
    for audience, keywords in AUDIENCE_KEYWORDS.items():
        if any(w in q for w in keywords):
            return audience
    return None  # No routing — search everything


class RAGService:
    def __init__(self, settings: AppSettings, vector_store: VectorStoreConnector, chunker: ChunkingStrategy):
        self._settings = settings
        self._vs = vector_store
        self._chunker = chunker
        # Choose enricher based on config
        if settings.retriever.enrichment_mode == "llm":
            self._enricher = MetadataEnricher(google_api_key=settings.gemini.api_key)
        else:
            self._enricher = ProductionEnricher()
        self._crag = CRAGEvaluator(google_api_key=settings.gemini.api_key) if settings.retriever.enable_crag else None
        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini.model, temperature=settings.gemini.temperature,
            max_output_tokens=settings.gemini.max_tokens, google_api_key=settings.gemini.api_key,
        )
        self._reranker = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", temperature=0, max_output_tokens=5, google_api_key=settings.gemini.api_key,
        )

    def ingest_pdf_pages(self, pages: list[Document], enrich_metadata: bool = True) -> int:
        chunks = self._chunker.chunk(pages)
        if enrich_metadata:
            chunks = self._enricher.enrich_batch(chunks)
        self._vs.index_documents(chunks)
        logger.info("ingested", pages=len(pages), chunks=len(chunks), enriched=enrich_metadata)
        return len(chunks)

    def _rerank(self, query: str, candidates: list[Document]) -> list[Document]:
        top_k = self._settings.retriever.rerank_top_k
        scored = []
        for d in candidates:
            try:
                score = int(self._reranker.invoke(
                    RERANK_PROMPT.format(query=query, chunk=d.page_content[:600])
                ).content.strip())
            except:
                score = 0
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]

    def _routed_retrieve(self, user_query: str, content_type_filter: str | None = None) -> tuple[list[Document], str | None]:
        """Retrieve with audience-based routing when confident, otherwise full search."""
        k = self._settings.retriever.initial_k

        # Explicit content_type filter takes priority
        if content_type_filter:
            return self._vs.filtered_search(user_query, "content_type", content_type_filter, k=k), None

        # Audience-based routing
        audience = detect_query_audience(user_query)
        if audience and audience in AUDIENCE_TO_DOCTYPE:
            target_dt = AUDIENCE_TO_DOCTYPE[audience]
            docs = self._vs.filtered_search(user_query, "doc_type", target_dt, k=k)
            if docs:
                logger.info("routed_retrieval", audience=audience, target=target_dt, results=len(docs))
                return docs, audience
            # Fall through to unfiltered if filtered returns nothing
            logger.info("routing_fallback", audience=audience, reason="no_filtered_results")

        return self._vs.search(user_query, k=k), audience

    def query(self, user_query: str, content_type_filter: str | None = None) -> dict:
        # Step 1: Retrieve (with routing)
        candidates, audience = self._routed_retrieve(user_query, content_type_filter)

        if not candidates:
            return {"answer": "No relevant documents found.", "sources": [], "verdict": "INCORRECT", "attempts": 1}

        # Step 2: CRAG evaluation
        verdict = "CORRECT"
        refined_query = None
        attempts = 1

        if self._crag:
            verdict = self._crag.evaluate(user_query, candidates)

            if verdict == "INCORRECT":
                return {
                    "answer": "I cannot answer this accurately based on available documents.",
                    "sources": [], "verdict": verdict, "attempts": 1,
                }

            if verdict == "AMBIGUOUS":
                refined_query = self._crag.refine_query(user_query, candidates)
                candidates = self._vs.search(refined_query, k=self._settings.retriever.initial_k)
                attempts = 2
                verdict = self._crag.evaluate(refined_query, candidates)
                if verdict == "INCORRECT":
                    return {
                        "answer": "After query refinement, still insufficient context.",
                        "sources": [], "verdict": verdict, "refined_query": refined_query, "attempts": 2,
                    }

        # Step 3: Rerank
        reranked = self._rerank(user_query, candidates)

        # Step 4: Generate with source citations
        ctx_parts = []
        sources = []
        for d in reranked:
            source_name = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", "?")
            ctype = d.metadata.get("content_type", "unknown")
            ctx_parts.append(f"[Source: {source_name}] (Page {page}, {ctype}):\n{d.page_content}")
            sources.append({
                "source": source_name, "page": page, "content_type": ctype,
                "topic": d.metadata.get("topic", ""),
                "audience": d.metadata.get("audience", ""),
                "snippet": d.page_content[:120],
            })

        response = self._llm.invoke(GROUNDED_PROMPT.format(context="\n\n".join(ctx_parts), question=user_query))

        return {
            "answer": response.content,
            "sources": sources,
            "verdict": verdict,
            "refined_query": refined_query,
            "attempts": attempts,
            "routed_audience": audience,
        }