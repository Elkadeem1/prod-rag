"""Chunking Strategies — Fixed, Recursive, Semantic, Markdown-Aware."""
from typing import Protocol
import re, structlog
import numpy as np
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_voyageai import VoyageAIEmbeddings

logger = structlog.get_logger()


class ChunkingStrategy(Protocol):
    def chunk(self, documents: list[Document]) -> list[Document]: ...


class FixedChunker:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 0):
        self._splitter = CharacterTextSplitter(separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    def chunk(self, documents: list[Document]) -> list[Document]:
        chunks = self._splitter.split_documents(documents)
        logger.info("chunked", strategy="fixed", count=len(chunks))
        return chunks


class RecursiveChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " "],
        )
    def chunk(self, documents: list[Document]) -> list[Document]:
        chunks = self._splitter.split_documents(documents)
        for c in chunks:
            fl = c.page_content.split("\n")[0].strip()
            c.metadata.update({
                "section": fl if len(fl) < 100 else "body",
                "char_count": len(c.page_content),
                "has_numbers": bool(re.search(r'\d+\.\d+', c.page_content)),
                "has_references": bool(re.search(r'\[\d+\]', c.page_content)),
            })
        logger.info("chunked", strategy="recursive", count=len(chunks))
        return chunks


class SemanticChunker:
    """Groups consecutive sentences by embedding cosine similarity."""

    def __init__(self, embeddings: VoyageAIEmbeddings, similarity_threshold: float = 0.72, max_chunk_chars: int = 800):
        self._embeddings = embeddings
        self._threshold = similarity_threshold
        self._max_chars = max_chunk_chars

    def chunk(self, documents: list[Document]) -> list[Document]:
        all_sents = []
        for doc in documents:
            sents = re.split(r'(?<=[.!?])\s+', doc.page_content)
            for s in sents:
                if s.strip() and len(s.strip()) > 10:
                    all_sents.append({"text": s.strip(), "page": doc.metadata.get("page", 0)})

        if not all_sents:
            return []

        # Embed all sentences
        texts = [s["text"] for s in all_sents]
        vecs = self._embeddings.embed_documents(texts)

        # Group by similarity
        chunks = []
        group = [all_sents[0]]
        gvecs = [vecs[0]]

        for i in range(1, len(all_sents)):
            centroid = np.mean(gvecs, axis=0)
            norm_c = np.linalg.norm(centroid)
            norm_v = np.linalg.norm(vecs[i])
            sim = np.dot(centroid, vecs[i]) / (norm_c * norm_v + 1e-8) if norm_c > 0 and norm_v > 0 else 0

            group_text = " ".join(s["text"] for s in group)
            if sim >= self._threshold and len(group_text) < self._max_chars:
                group.append(all_sents[i])
                gvecs.append(vecs[i])
            else:
                pages = list(set(s["page"] for s in group))
                chunks.append(Document(
                    page_content=group_text,
                    metadata={"page": pages[0], "pages": pages, "strategy": "semantic", "char_count": len(group_text)},
                ))
                group = [all_sents[i]]
                gvecs = [vecs[i]]

        if group:
            chunks.append(Document(
                page_content=" ".join(s["text"] for s in group),
                metadata={"page": group[0]["page"], "strategy": "semantic"},
            ))

        logger.info("chunked", strategy="semantic", count=len(chunks), sentences=len(all_sents))
        return chunks


class MarkdownAwareChunker:
    """For mixed corpora: split markdown by headers, PDFs by recursive."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        md_pages = [p for p in documents if p.metadata.get("format") == "markdown"]
        pdf_pages = [p for p in documents if p.metadata.get("format") != "markdown"]

        chunks = []

        # Markdown: split by headers first, then recursive within sections
        for p in md_pages:
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("##", "section"), ("###", "subsection")],
            )
            md_chunks = header_splitter.split_text(p.page_content)
            for mc in md_chunks:
                mc.metadata.update(p.metadata)
                mc.metadata["strategy"] = "markdown"
            # Sub-split large sections
            sub_chunks = self._sub_splitter.split_documents(md_chunks)
            chunks.extend(sub_chunks)

        # PDF: fall back to recursive
        if pdf_pages:
            r_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap,
            )
            pdf_chunks = r_splitter.split_documents(pdf_pages)
            for c in pdf_chunks:
                c.metadata["strategy"] = "markdown"
            chunks.extend(pdf_chunks)

        logger.info("chunked", strategy="markdown", count=len(chunks))
        return chunks


def get_chunking_strategy(
    name: str, embeddings: VoyageAIEmbeddings | None = None,
    chunk_size: int = 500, chunk_overlap: int = 50, semantic_threshold: float = 0.72,
) -> ChunkingStrategy:
    if name == "fixed":
        return FixedChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif name == "recursive":
        return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif name == "semantic":
        if not embeddings:
            raise ValueError("Semantic chunker requires embeddings instance")
        return SemanticChunker(embeddings=embeddings, similarity_threshold=semantic_threshold)
    elif name == "markdown":
        return MarkdownAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raise ValueError(f"Unknown strategy: {name}")