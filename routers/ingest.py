"""Ingest Router — PDF and Markdown upload with single or bulk file support."""
import re, time, structlog, fitz
from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Depends
from pydantic import BaseModel
from langchain.schema import Document
from services.rag_service import RAGService
from services.metrics import rag_requests_total, rag_request_duration_seconds, rag_chunks_created_total
from dependencies import get_rag_service

logger = structlog.get_logger()
router = APIRouter(tags=["Ingestion"])


class FileResult(BaseModel):
    filename: str
    pages_parsed: int
    chunks_created: int
    doc_type: str
    error: str | None = None

class IngestResponse(BaseModel):
    status: str
    total_files: int
    total_chunks: int
    metadata_enriched: bool
    files: list[FileResult]


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    file: list[UploadFile] = File(..., description="One or more PDF/Markdown files"),
    enrich: bool = Query(True, description="Run metadata extraction"),
    svc: RAGService = Depends(get_rag_service),
):
    """Ingest one or multiple files in a single request.

    Supports PDF (.pdf) and Markdown (.md) files.
    Upload multiple files by selecting them all in the file picker.
    """
    t0 = time.time()
    if not file:
        raise HTTPException(400, "At least one file is required.")

    results = []
    total_chunks = 0

    for f in file:
        result = await _process_single_file(f, enrich, svc)
        results.append(result)
        if result.error is None:
            total_chunks += result.chunks_created
            # Record chunks created metric
            strategy = svc._settings.chunking.strategy if hasattr(svc, '_settings') else "recursive"
            rag_chunks_created_total.labels(strategy=strategy).inc(result.chunks_created)

    status = "success" if all(r.error is None for r in results) else "partial"
    if all(r.error is not None for r in results):
        status = "failed"

    # Record request metrics
    rag_requests_total.labels(endpoint="/ingest", status="200").inc()
    rag_request_duration_seconds.labels(endpoint="/ingest").observe(time.time() - t0)

    return IngestResponse(
        status=status,
        total_files=len(results),
        total_chunks=total_chunks,
        metadata_enriched=enrich,
        files=results,
    )


async def _process_single_file(
    file: UploadFile, enrich: bool, svc: RAGService
) -> FileResult:
    """Process a single uploaded file and return its result."""
    if not file.filename:
        return FileResult(filename="unknown", pages_parsed=0, chunks_created=0,
                          doc_type="unknown", error="Filename required")

    fname = file.filename.lower()
    if not (fname.endswith(".pdf") or fname.endswith(".md")):
        return FileResult(filename=file.filename, pages_parsed=0, chunks_created=0,
                          doc_type="unknown", error="Unsupported format. Only PDF and Markdown (.md) files accepted.")

    try:
        content = await file.read()

        if fname.endswith(".pdf"):
            pages = _parse_pdf(content, file.filename)
            doc_type = "academic_paper"
        else:
            pages = _parse_markdown(content.decode("utf-8"), file.filename)
            doc_type = _infer_doc_type(file.filename)

        if not pages:
            return FileResult(filename=file.filename, pages_parsed=0, chunks_created=0,
                              doc_type=doc_type, error="No extractable text")

        chunks = svc.ingest_pdf_pages(pages, enrich_metadata=enrich)
        logger.info("file_ingested", filename=file.filename, pages=len(pages),
                     chunks=chunks, doc_type=doc_type)

        return FileResult(
            filename=file.filename, pages_parsed=len(pages),
            chunks_created=chunks, doc_type=doc_type,
        )
    except Exception as e:
        logger.error("file_ingest_failed", filename=file.filename, error=str(e))
        return FileResult(filename=file.filename, pages_parsed=0, chunks_created=0,
                          doc_type="unknown", error=str(e))


def _parse_pdf(content: bytes, filename: str) -> list[Document]:
    """Parse PDF into page-level documents."""
    pdf = fitz.open(stream=content, filetype="pdf")
    pages = [
        Document(
            page_content=pdf[i].get_text("text"),
            metadata={"source": filename, "page": i + 1, "total_pages": len(pdf),
                       "doc_type": "academic_paper", "format": "pdf"},
        )
        for i in range(len(pdf))
        if pdf[i].get_text("text").strip()
    ]
    pdf.close()
    return pages


def _parse_markdown(content: str, filename: str) -> list[Document]:
    """Parse Markdown into section-level documents (split by ## headers)."""
    doc_type = _infer_doc_type(filename)
    sections = re.split(r'\n(?=## )', content)
    pages = [
        Document(
            page_content=s.strip(),
            metadata={"source": filename, "page": i + 1,
                       "doc_type": doc_type, "format": "markdown"},
        )
        for i, s in enumerate(sections)
        if s.strip()
    ]
    return pages


def _infer_doc_type(filename: str) -> str:
    """Infer doc_type from filename patterns."""
    name = filename.lower()
    if "faq" in name:
        return "customer_faq"
    if "wiki" in name:
        return "internal_wiki"
    return "markdown_doc"