#  prod-rag — Production RAG API with CRAG & Precision Chunking

A production-ready **Retrieval-Augmented Generation (RAG)** API built with **FastAPI**, featuring Corrective RAG (CRAG), multiple intelligent chunking strategies, metadata enrichment, query routing, and a full observability stack — all deployable via Docker Compose.

---

##  Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
│                                                                  │
│  /ingest ──► ChunkingStrategy ──► MetadataEnricher ──► Qdrant   │
│                                                                  │
│  /chat ───► QueryRouter ──► VectorSearch ──► CRAGEvaluator       │
│                  │                               │               │
│                  └──► Reranker (Gemini) ──► LLM Answer           │
│                                                                  │
│  /compare ─► Side-by-side strategy benchmarking                 │
└─────────────────────────────────────────────────────────────────┘
         │                          │
    Prometheus ◄──────────────── /metrics
         │
      Grafana Dashboard
```

---

##  Features

###  Chunking Strategies (4 modes)
| Strategy | Description |
|----------|-------------|
| `fixed` | Character-level fixed-size splitting |
| `recursive` | Hierarchical splitting on `\n\n`, `\n`, `.`, `,` with rich metadata |
| `semantic` | Sentence-level grouping by VoyageAI embedding cosine similarity |
| `markdown` | Header-aware splitting for `.md` files + recursive fallback for PDFs |

###  Corrective RAG (CRAG)
- Evaluates retrieved context quality before generating an answer
- Returns a **verdict**: `CORRECT`, `AMBIGUOUS`, or `INCORRECT`
- On `AMBIGUOUS`: automatically **refines the query** and re-retrieves
- On `INCORRECT`: refuses to hallucinate an answer

###  Metadata Enrichment (2 modes)
| Mode | Method | Cost |
|------|--------|------|
| `regex` | Deterministic pattern matching (content type, audience, complexity, entities) | Free |
| `llm` | Gemini-powered semantic extraction (topic, audience, key concepts) | ~$0.004/chunk |

###  Query Audience Routing
Automatically classifies queries into audiences and routes them to matching document types:
- `technical` → `internal_wiki`
- `academic` → `academic_paper`
- `general` → `customer_faq`

Falls back to full collection search if no routed results are found.

###  Observability
- **Prometheus** metrics scraping from `/metrics`
- **Grafana** dashboards pre-provisioned
- Metrics tracked: request counts, latency, CRAG verdicts, chunking duration, enrichment duration, chunks created

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI + Uvicorn |
| LLM | Google Gemini 2.0 Flash |
| Embeddings | VoyageAI (`voyage-3`) |
| Vector Store | Qdrant |
| PDF Parsing | PyMuPDF (fitz) |
| Logging | structlog (JSON in prod, pretty in dev) |
| Metrics | Prometheus Client |
| Containerization | Docker + Docker Compose |
| Load Testing | Grafana k6 |

---

##  Getting Started

### Prerequisites
- Docker & Docker Compose installed
- API keys for **Google Gemini** and **VoyageAI**

### 1. Clone the repository
```bash
git clone https://github.com/Elkadeem1/prod-rag.git
cd prod-rag
```

### 2. Set up environment variables
Create a `.env` file in the project root:
```env
# Google Gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash
GEMINI_TEMPERATURE=0.0
GEMINI_MAX_TOKENS=2048

# VoyageAI Embeddings
VOYAGE_API_KEY=your_voyage_api_key
VOYAGE_MODEL=voyage-3

# Qdrant Vector Store
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=s2_enriched_rag

# Chunking
CHUNKING_STRATEGY=recursive       # fixed | recursive | semantic | markdown
CHUNKING_CHUNK_SIZE=500
CHUNKING_CHUNK_OVERLAP=50
CHUNKING_SEMANTIC_THRESHOLD=0.72  # only used when strategy=semantic

# Retriever
RETRIEVER_INITIAL_K=10
RETRIEVER_RERANK_TOP_K=3
RETRIEVER_ENABLE_CRAG=true
RETRIEVER_ENRICHMENT_MODE=regex   # regex | llm
```

### 3. Start all services
```bash
docker-compose up --build
```

This starts:
- **RAG API** → `http://localhost:8000`
- **Qdrant** → `http://localhost:6333`
- **Prometheus** → `http://localhost:9090`
- **Grafana** → `http://localhost:3000` (admin / admin)

### 4. Verify the API is healthy
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "qdrant": true,
  "version": "2.0.0",
  "chunking_strategy": "recursive",
  "crag_enabled": true,
  "enrichment_mode": "regex"
}
```

---

## 📡 API Endpoints

### `POST /api/v1/ingest`
Upload one or more PDF or Markdown files for ingestion.

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@paper.pdf" \
  -F "file=@wiki.md" \
  -F "enrich=true"
```

**Response:**
```json
{
  "status": "success",
  "total_files": 2,
  "total_chunks": 87,
  "metadata_enriched": true,
  "files": [
    { "filename": "paper.pdf", "pages_parsed": 12, "chunks_created": 64, "doc_type": "academic_paper" },
    { "filename": "wiki.md",   "pages_parsed": 5,  "chunks_created": 23, "doc_type": "internal_wiki" }
  ]
}
```

---

### `POST /api/v1/chat`
Ask a question against the ingested documents.

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How does the defense mechanism work?"}'
```

**Response:**
```json
{
  "query": "How does the defense mechanism work?",
  "answer": "According to [Source: paper.pdf], the defense mechanism...",
  "sources": [
    {
      "source": "paper.pdf",
      "page": 4,
      "content_type": "methodology",
      "topic": "adversarial defense",
      "snippet": "The proposed mechanism..."
    }
  ],
  "verdict": "CORRECT",
  "refined_query": null,
  "attempts": 1,
  "routed_audience": "academic"
}
```

You can also filter by content type:
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the results?", "content_type_filter": "results"}'
```

---

### `POST /api/v1/compare/chunking`
Benchmark all chunking strategies on the same text.

```bash
curl -X POST http://localhost:8000/api/v1/compare/chunking \
  -H "Content-Type: application/json" \
  -d '{"text": "Your long text here...", "chunk_size": 500, "format": "text"}'
```

---

### `POST /api/v1/compare/enrichment`
Compare regex vs LLM metadata enrichment side by side.

```bash
curl -X POST http://localhost:8000/api/v1/compare/enrichment \
  -H "Content-Type: application/json" \
  -d '{"text": "In this study, we propose a novel architecture..."}'
```

---

### `POST /api/v1/compare/retrieval`
Compare naive search, audience-routed search, and full CRAG pipeline.

---

### `GET /health` · `GET /metrics`
System health check and raw Prometheus metrics.

---

##  Running Tests

### Install test dependencies
```bash
pip install -r tests/requirements-test.txt
```

### Unit tests
```bash
pytest tests/unit/
```

### Integration tests
```bash
pytest tests/integration/
```

### Load tests (k6)
```bash
docker-compose --profile testing up k6
```

---

##  Project Structure

```
prod-rag/
├── main.py                    # FastAPI app + lifespan
├── config.py                  # Pydantic settings (env-driven)
├── dependencies.py            # DI providers for RAGService & VectorStore
├── middleware.py              # Request ID injection middleware
│
├── routers/
│   ├── chat.py                # POST /api/v1/chat
│   ├── ingest.py              # POST /api/v1/ingest
│   ├── compare.py             # POST /api/v1/compare/*
│   └── debug.py               # Debug/inspection endpoints
│
├── services/
│   ├── rag_service.py         # Orchestration: retrieve → CRAG → rerank → generate
│   ├── chunking.py            # Fixed, Recursive, Semantic, Markdown chunkers
│   ├── crag.py                # CRAGEvaluator: evaluate + refine_query
│   ├── metadata.py            # ProductionEnricher (regex) & MetadataEnricher (LLM)
│   └── metrics.py             # Prometheus counters/histograms
│
├── database/
│   └── vector_store.py        # QdrantConnector: index, search, filtered_search
│
├── monitoring/
│   ├── prometheus.yml         # Scrape config
│   └── grafana/               # Dashboard JSON + provisioning
│
├── tests/
│   ├── unit/                  # Chunking, CRAG, metadata, routing tests
│   ├── integration/           # Full API tests
│   └── load/                  # k6 smoke test
│
├── postman/
│   └── rag_api.postman_collection.json   # Ready-to-import Postman collection
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

##  Configuration Reference

All settings are loaded from `.env` via Pydantic Settings. Every setting can also be overridden as an environment variable.

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model for generation & reranking |
| `GEMINI_TEMPERATURE` | `0.0` | LLM temperature |
| `VOYAGE_MODEL` | `voyage-3` | Embedding model |
| `QDRANT_COLLECTION_NAME` | `s2_enriched_rag` | Vector collection name |
| `CHUNKING_STRATEGY` | `recursive` | `fixed` / `recursive` / `semantic` / `markdown` |
| `CHUNKING_CHUNK_SIZE` | `500` | Target chunk size in characters |
| `CHUNKING_SEMANTIC_THRESHOLD` | `0.72` | Cosine similarity threshold (semantic mode only) |
| `RETRIEVER_INITIAL_K` | `10` | Candidates retrieved before reranking |
| `RETRIEVER_RERANK_TOP_K` | `3` | Top-K kept after reranking |
| `RETRIEVER_ENABLE_CRAG` | `true` | Toggle CRAG self-correction |
| `RETRIEVER_ENRICHMENT_MODE` | `regex` | `regex` (free) or `llm` (accurate) |

---

##  RAG Query Pipeline

```
User Query
    │
    ▼
Audience Detection  ──► route to doc_type filter (optional)
    │
    ▼
Vector Search (Qdrant)  ──  initial_k = 10 candidates
    │
    ▼
CRAG Evaluation  ──► CORRECT → continue
                 ──► AMBIGUOUS → refine query → re-search → re-evaluate
                 ──► INCORRECT → return "cannot answer"
    │
    ▼
LLM Reranker (Gemini)  ──  score each chunk 0-10, keep top_k = 3
    │
    ▼
Answer Generation (Gemini)  ──  grounded prompt with [Source: filename] citations
    │
    ▼
ChatResponse { answer, sources, verdict, attempts, routed_audience }
```

---

##  Postman Collection

A complete Postman collection covering all endpoints is included at:
```
postman/rag_api.postman_collection.json
```

Import it into Postman and set a `base_url` variable to `http://localhost:8000`.

---

##  License

This project is licensed under the terms of the [LICENSE](./LICENSE) file included in the repository.
