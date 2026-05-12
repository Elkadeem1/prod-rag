"""Corrective RAG (CRAG) — Self-evaluating retrieval pipeline."""
import structlog
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI

logger = structlog.get_logger()


class CRAGEvaluator:
    """
    Evaluates retrieval quality and decides the generation strategy:
    - CORRECT: generate answer from context
    - AMBIGUOUS: refine query, re-retrieve, then generate
    - INCORRECT: refuse to answer
    """

    def __init__(self, google_api_key: str, model: str = "gemini-2.0-flash"):
        self._llm = ChatGoogleGenerativeAI(
            model=model, temperature=0, max_output_tokens=50, google_api_key=google_api_key,
        )

    def evaluate(self, query: str, docs: list[Document]) -> str:
        ctx = "\n".join(d.page_content[:300] for d in docs[:3])
        prompt = f"""Evaluate: can the following context answer the query?
Return ONLY one word: CORRECT, AMBIGUOUS, or INCORRECT.

Query: {query}
Context: {ctx[:2000]}
Verdict:"""
        verdict = self._llm.invoke(prompt).content.strip().upper()
        # Normalize
        for v in ("CORRECT", "AMBIGUOUS", "INCORRECT"):
            if v in verdict:
                logger.info("crag_verdict", query=query[:50], verdict=v)
                return v
        logger.warning("crag_parse_fallback", raw=verdict)
        return "AMBIGUOUS"

    def refine_query(self, original_query: str, docs: list[Document]) -> str:
        ctx = "\n".join(d.page_content[:200] for d in docs[:2])
        prompt = f"""The context was partially relevant. Generate a more specific search query.
Return ONLY the refined query.

Original: {original_query}
Context: {ctx[:1000]}
Refined:"""
        refined = self._llm.invoke(prompt).content.strip()
        logger.info("crag_refined", original=original_query[:50], refined=refined[:50])
        return refined