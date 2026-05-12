"""Unit tests for CRAG evaluator logic."""
import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document
from services.crag import CRAGEvaluator


class TestCRAGVerdictParsing:
    """Test that CRAG correctly parses LLM verdicts."""

    @pytest.fixture
    def evaluator(self):
        with patch.object(CRAGEvaluator, '__init__', lambda self, **kwargs: None):
            e = CRAGEvaluator.__new__(CRAGEvaluator)
            e._llm = MagicMock()
            return e

    @pytest.mark.parametrize("response, expected", [
        ("CORRECT", "CORRECT"),
        ("AMBIGUOUS", "AMBIGUOUS"),
        ("INCORRECT", "INCORRECT"),
        ("The verdict is CORRECT.", "CORRECT"),
        ("Based on analysis: AMBIGUOUS", "AMBIGUOUS"),
    ])
    def test_verdict_extraction(self, evaluator, response, expected):
        evaluator._llm.invoke.return_value = MagicMock(content=response)
        docs = [Document(page_content="test content", metadata={"page": 1})]
        assert evaluator.evaluate("test query", docs) == expected

    def test_fallback_to_ambiguous(self, evaluator):
        evaluator._llm.invoke.return_value = MagicMock(content="I'm not sure about this")
        docs = [Document(page_content="test", metadata={"page": 1})]
        assert evaluator.evaluate("test", docs) == "AMBIGUOUS"