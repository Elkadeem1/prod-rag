"""Unit tests for query audience routing."""
import pytest
from services.rag_service import detect_query_audience


class TestQueryRouting:
    """Test audience detection from NB02 Section 4."""

    def test_technical_queries(self):
        assert detect_query_audience("What is the P50 latency budget for Tier 0 deployments?") == "technical"
        assert detect_query_audience("How should we configure the production architecture?") == "technical"
        assert detect_query_audience("What is the deployment cost?") == "technical"

    def test_academic_queries(self):
        assert detect_query_audience("What attack methods were evaluated in the study?") == "academic"
        assert detect_query_audience("Smith et al found that the experiment showed...") == "academic"
        assert detect_query_audience("What does the research paper conclude?") == "academic"

    def test_general_queries(self):
        assert detect_query_audience("How do I stop hackers from messing with my chatbot?") == "general"
        assert detect_query_audience("Can I fix this simple issue?") == "general"
        assert detect_query_audience("What is prompt injection?") == "general"

    def test_unroutable_query(self):
        """Queries without clear audience signals should return None (search all)."""
        assert detect_query_audience("Compare regex and LLM-based input filtering") is None