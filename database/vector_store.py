import time, structlog
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from langchain_voyageai import VoyageAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.schema import Document
from config import QdrantSettings, VoyageSettings

logger = structlog.get_logger()
VOYAGE_3_DIM = 1024


class VectorStoreConnector:
    def __init__(self, qdrant: QdrantSettings, voyage: VoyageSettings):
        self._settings = qdrant
        self._client = QdrantClient(url=qdrant.url, api_key=qdrant.api_key, timeout=30)
        self._embeddings = VoyageAIEmbeddings(model=voyage.model, voyage_api_key=voyage.api_key)
        self._vs: QdrantVectorStore | None = None

    @property
    def embeddings(self) -> VoyageAIEmbeddings:
        return self._embeddings

    def health_check(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    def ensure_collection(self):
        name = self._settings.collection_name
        if not self._client.collection_exists(name):
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VOYAGE_3_DIM, distance=Distance.COSINE),
            )
        self._vs = QdrantVectorStore(client=self._client, collection_name=name, embedding=self._embeddings)

    def index_documents(self, documents: list[Document], max_retries: int = 3) -> int:
        if not self._vs: self.ensure_collection()
        for attempt in range(max_retries):
            try:
                self._vs.add_documents(documents)
                logger.info("indexed", count=len(documents))
                return len(documents)
            except (ResponseHandlingException, ConnectionError) as e:
                time.sleep(2 ** attempt)
        raise RuntimeError("Index failed")

    def search(self, query: str, k: int = 10) -> list[Document]:
        if not self._vs: raise RuntimeError("Not initialized")
        return self._vs.similarity_search(query, k=k)

    def filtered_search(self, query: str, field: str, value: str, k: int = 5) -> list[Document]:
        """Search with Qdrant payload filter — narrows search before similarity."""
        query_vec = self._embeddings.embed_query(query)
        results = self._client.query_points(
            collection_name=self._settings.collection_name,
            query=query_vec,
            query_filter=Filter(must=[FieldCondition(key=f"metadata.{field}", match=MatchValue(value=value))]),
            limit=k,
        )
        docs = []
        for pt in results.points:
            pl = pt.payload or {}
            docs.append(Document(
                page_content=pl.get("page_content", ""),
                metadata={**pl.get("metadata", {}), "score": pt.score},
            ))
        return docs

    def drop_collection(self):
        if self._client.collection_exists(self._settings.collection_name):
            self._client.delete_collection(self._settings.collection_name)
            self._vs = None

    @property
    def is_ready(self) -> bool:
        return self._vs is not None