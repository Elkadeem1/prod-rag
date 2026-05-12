from pydantic import Field  
from pydantic_settings import BaseSettings


class GeminiSettings(BaseSettings):
    model: str = Field(default="gemini-2.0-flash")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    api_key: str = Field(description="Google API key")
    model_config = {"env_prefix": "GEMINI_"}


class VoyageSettings(BaseSettings):
    model: str = Field(default="voyage-3")
    api_key: str = Field(description="Voyage AI API key")
    model_config = {"env_prefix": "VOYAGE_"}


class QdrantSettings(BaseSettings):
    url: str = Field(default="http://localhost:6333")
    api_key: str | None = Field(default=None)
    collection_name: str = Field(default="s2_enriched_rag")
    model_config = {"env_prefix": "QDRANT_"}


class ChunkingSettings(BaseSettings):
    strategy: str = Field(default="recursive", description="fixed | recursive | semantic | markdown")
    chunk_size: int = Field(default=500, gt=50)
    chunk_overlap: int = Field(default=50, ge=0)
    semantic_threshold: float = Field(default=0.72, ge=0.0, le=1.0, description="Cosine similarity threshold for semantic chunking")
    model_config = {"env_prefix": "CHUNKING_"}


class RetrieverSettings(BaseSettings):
    initial_k: int = Field(default=10, gt=0)
    rerank_top_k: int = Field(default=3, gt=0)
    enable_crag: bool = Field(default=True, description="Enable Corrective RAG evaluation")
    enrichment_mode: str = Field(default="regex", description="regex (free, deterministic) | llm (accurate, costs tokens)")
    model_config = {"env_prefix": "RETRIEVER_"}


class AppSettings(BaseSettings):
    app_name: str = "Precision Chunking & Metadata Enrichment API"
    version: str = "2.0.0"
    debug: bool = False
    log_level: str = "INFO"
    gemini: GeminiSettings = GeminiSettings()
    voyage: VoyageSettings = VoyageSettings()
    qdrant: QdrantSettings = QdrantSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    retriever: RetrieverSettings = RetrieverSettings()
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def get_settings() -> AppSettings:
    return AppSettings()