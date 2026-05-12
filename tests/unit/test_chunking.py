"""Unit tests for chunking strategies."""
import pytest
import numpy as np
from unittest.mock import MagicMock
from langchain.schema import Document
from services.chunking import FixedChunker, RecursiveChunker, SemanticChunker, MarkdownAwareChunker, get_chunking_strategy

DOCS = [
    Document(page_content="Machine learning is transforming healthcare. Deep learning models can detect diseases from medical images. This has significant implications for early diagnosis.", metadata={"page": 1}),
    Document(page_content="The methodology involves training a CNN on 10,000 X-ray images. We used a learning rate of 0.001 and batch size of 32. Results show 95.2% accuracy [1].", metadata={"page": 2}),
]

MD_DOCS = [
    Document(
        page_content="## Overview\nThis is the overview section.\n\n## Methods\nWe used advanced techniques.\n\n### Sub-method A\nDetails about sub-method A.\n\n## Results\nThe results were significant with 95.2% accuracy.",
        metadata={"page": 1, "format": "markdown", "doc_type": "internal_wiki", "source": "test.md"},
    ),
]

MIXED_DOCS = [
    Document(page_content="Academic paper content about neural networks and deep learning architectures.", metadata={"page": 1, "format": "pdf", "doc_type": "academic_paper"}),
    Document(
        page_content="## FAQ Section\nHow do I use the API?\n\n## Troubleshooting\nCommon issues and solutions.",
        metadata={"page": 1, "format": "markdown", "doc_type": "customer_faq", "source": "faq.md"},
    ),
]

class TestFixedChunker:
    def test_chunks_produced(self):
        assert len(FixedChunker(50).chunk(DOCS)) > 2

    def test_size_limit(self):
        for c in FixedChunker(50).chunk(DOCS):
            assert len(c.page_content) <= 50

class TestRecursiveChunker:
    def test_metadata_enrichment(self):
        chunks = RecursiveChunker(100, 10).chunk(DOCS)
        for c in chunks:
            assert "section" in c.metadata
            assert "char_count" in c.metadata

    def test_numeric_detection(self):
        chunks = RecursiveChunker(200).chunk(DOCS)
        assert any(c.metadata.get("has_numbers") for c in chunks)

    def test_reference_detection(self):
        chunks = RecursiveChunker(200).chunk(DOCS)
        assert any(c.metadata.get("has_references") for c in chunks)

class TestSemanticChunker:
    def test_with_mock_embeddings(self):
        mock_emb = MagicMock()
        # Return distinct vectors for different topics
        mock_emb.embed_documents.return_value = [
            [1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0],  # Similar group
            [0.0, 0.0, 1.0], [0.1, 0.0, 0.9], [0.0, 0.1, 0.8],  # Different group
        ]
        chunker = SemanticChunker(embeddings=mock_emb, similarity_threshold=0.7)
        chunks = chunker.chunk(DOCS)
        assert len(chunks) >= 2  # Should create at least 2 groups

    def test_filters_short_sentences(self):
        """Sentences <= 10 chars should be filtered out (NB01 behavior)."""
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[1.0, 0.0]]
        short_doc = [Document(page_content="Hi. Ok. Done. This sentence is long enough to pass.", metadata={"page": 1})]
        chunker = SemanticChunker(embeddings=mock_emb, similarity_threshold=0.5)
        chunks = chunker.chunk(short_doc)
        # Only the long sentence should pass the filter
        for c in chunks:
            assert len(c.page_content) > 10

class TestMarkdownAwareChunker:
    def test_splits_by_headers(self):
        chunker = MarkdownAwareChunker(chunk_size=500)
        chunks = chunker.chunk(MD_DOCS)
        assert len(chunks) >= 2  # Should split by ## headers

    def test_preserves_metadata(self):
        chunker = MarkdownAwareChunker(chunk_size=500)
        chunks = chunker.chunk(MD_DOCS)
        for c in chunks:
            assert "strategy" in c.metadata

    def test_mixed_corpus_handling(self):
        """Markdown docs → header split, PDF docs → recursive split."""
        chunker = MarkdownAwareChunker(chunk_size=500)
        chunks = chunker.chunk(MIXED_DOCS)
        assert len(chunks) >= 2  # At least one from each doc type
        strategies = [c.metadata.get("strategy") for c in chunks]
        assert all(s == "markdown" for s in strategies)

    def test_pdf_fallback(self):
        """Non-markdown docs should fall back to recursive."""
        pdf_only = [Document(page_content="A long text " * 100, metadata={"page": 1, "format": "pdf"})]
        chunker = MarkdownAwareChunker(chunk_size=100)
        chunks = chunker.chunk(pdf_only)
        assert len(chunks) >= 2  # Should be split by recursive

class TestFactory:
    def test_fixed(self):
        assert isinstance(get_chunking_strategy("fixed"), FixedChunker)

    def test_recursive(self):
        assert isinstance(get_chunking_strategy("recursive"), RecursiveChunker)

    def test_semantic_requires_embeddings(self):
        with pytest.raises(ValueError, match="embeddings"):
            get_chunking_strategy("semantic")

    def test_markdown(self):
        assert isinstance(get_chunking_strategy("markdown"), MarkdownAwareChunker)

    def test_invalid(self):
        with pytest.raises(ValueError):
            get_chunking_strategy("unknown")