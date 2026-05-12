"""Metadata Enrichment — Production (regex) and LLM-powered extraction."""
import json, re, structlog
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI

logger = structlog.get_logger()

# ── Regex-based content classification patterns ──────────────────────────────
CONTENT_PATTERNS = {
    'methodology': ['method', 'approach', 'technique', 'algorithm', 'procedure', 'framework', 'architecture'],
    'results': ['result', 'finding', 'performance', 'accuracy', 'score', 'benchmark', 'evaluation', 'ASV'],
    'defense': ['defense', 'protection', 'mitigation', 'prevention', 'security', 'filter', 'validation'],
    'attack': ['attack', 'injection', 'adversarial', 'exploit', 'vulnerability', 'hack', 'trick'],
    'background': ['introduction', 'overview', 'background', 'related work', 'prior', 'existing'],
    'practical': ['how to', 'step', 'guide', 'FAQ', 'example', 'tip', 'recommendation'],
}

AUDIENCE_SIGNALS = {
    'technical': ['implementation', 'API', 'deployment', 'latency', 'throughput', 'architecture', 'tier'],
    'academic': ['et al', 'hypothesis', 'experiment', r'Table \d', r'Figure \d', r'Section \d', 'Appendix'],
    'general': ['basically', 'imagine', 'think of', 'like', 'simple', 'easy', 'just'],
}


def classify_content(text: str) -> str:
    """Keyword-based content type classification — $0, deterministic."""
    scores = {}
    lower = text.lower()
    for ctype, keywords in CONTENT_PATTERNS.items():
        scores[ctype] = sum(1 for k in keywords if k.lower() in lower)
    return max(scores, key=scores.get) if any(scores.values()) else 'other'


def classify_audience(text: str) -> str:
    """Audience detection from vocabulary signals."""
    scores = {}
    for aud, patterns in AUDIENCE_SIGNALS.items():
        scores[aud] = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
    return max(scores, key=scores.get) if any(scores.values()) else 'general'


def extract_entities(text: str) -> list[str]:
    """Regex-based named entity extraction."""
    entities = []
    entities.extend(re.findall(r'[A-Z][a-z]+ (?:[A-Z][a-z]+ )*[A-Z][a-z]+', text)[:5])
    entities.extend(re.findall(r'\b[A-Z]{2,}\b', text)[:3])
    return list(set(entities))[:5]


def compute_complexity(text: str) -> str:
    """Heuristic complexity scoring based on word length, formulas, tables, numbers."""
    words = text.split()
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    has_formulas = bool(re.search(r'[=<>]|\\frac|\$.*\$', text))
    has_tables = bool(re.search(r'Table \d|\|.*\|', text))
    num_density = len(re.findall(r'\d+\.\d+', text)) / max(len(words), 1)
    score = avg_word_len * 0.3 + (1 if has_formulas else 0) * 2 + (1 if has_tables else 0) + num_density * 10
    if score > 4:
        return 'expert'
    if score > 2:
        return 'intermediate'
    return 'beginner'


class ProductionEnricher:
    """Zero-cost, deterministic enrichment using regex + heuristics.

    From NB02 Section 1: the recommended production approach.
    Adds: content_type, audience, complexity, key_entities,
          has_numbers, has_table, has_code, word_count.
    """

    def enrich(self, chunk: Document) -> Document:
        text = chunk.page_content
        chunk.metadata['content_type'] = classify_content(text)
        chunk.metadata['audience'] = classify_audience(text)
        chunk.metadata['complexity'] = compute_complexity(text)
        chunk.metadata['key_entities'] = extract_entities(text)
        chunk.metadata['has_numbers'] = bool(re.search(r'\d+\.\d+', text))
        chunk.metadata['has_table'] = bool(re.search(r'Table \d|\|.*\|.*\|', text))
        chunk.metadata['has_code'] = bool(re.search(r'```|def |class |import ', text))
        chunk.metadata['word_count'] = len(text.split())
        return chunk

    def enrich_batch(self, chunks: list[Document]) -> list[Document]:
        enriched = [self.enrich(c) for c in chunks]
        logger.info("production_enriched", count=len(enriched))
        return enriched


# ── LLM-powered enrichment (higher accuracy, costs API tokens) ───────────────

EXTRACT_PROMPT = """Analyze this text and extract metadata.
Return ONLY valid JSON (no markdown):
{{"topic": "<main topic>", "content_type": "<methodology|results|background|conclusion|reference|other>",
  "key_entities": ["<entity1>", "<entity2>"], "has_data": <true/false>, "summary": "<one sentence>"}}

Text:
{text}"""


class MetadataEnricher:
    """Uses Gemini to extract structured metadata from each chunk."""

    def __init__(self, google_api_key: str, model: str = "gemini-2.0-flash"):
        self._llm = ChatGoogleGenerativeAI(
            model=model, temperature=0, max_output_tokens=200, google_api_key=google_api_key,
        )

    def enrich(self, chunk: Document) -> Document:
        try:
            resp = self._llm.invoke(EXTRACT_PROMPT.format(text=chunk.page_content[:1500])).content.strip()
            if resp.startswith("```"):
                resp = resp.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            meta = json.loads(resp)
            chunk.metadata.update(meta)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("metadata_extraction_failed", error=str(e))
            chunk.metadata.update({"topic": "unknown", "content_type": "other", "key_entities": [], "has_data": False, "summary": ""})
        return chunk

    def enrich_batch(self, chunks: list[Document]) -> list[Document]:
        enriched = [self.enrich(c) for c in chunks]
        logger.info("metadata_enriched", count=len(enriched))
        return enriched