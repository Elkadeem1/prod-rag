"""Dependency Injection — FastAPI Depends() wiring."""
from functools import lru_cache
from config import get_settings
from database.vector_store import VectorStoreConnector
from services.chunking import get_chunking_strategy
from services.rag_service import RAGService

@lru_cache()
def get_vector_store():
    s = get_settings()
    c = VectorStoreConnector(qdrant=s.qdrant, voyage=s.voyage)
    c.ensure_collection()
    return c

@lru_cache()
def get_chunker():
    s = get_settings()
    vs = get_vector_store()
    return get_chunking_strategy(
        name=s.chunking.strategy, embeddings=vs.embeddings,
        chunk_size=s.chunking.chunk_size, chunk_overlap=s.chunking.chunk_overlap,
        semantic_threshold=s.chunking.semantic_threshold,
    )

@lru_cache()
def get_rag_service():
    return RAGService(settings=get_settings(), vector_store=get_vector_store(), chunker=get_chunker())