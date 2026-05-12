"""Unit tests for metadata enrichment — Production (regex) enricher."""
import pytest
from langchain.schema import Document
from services.metadata import ProductionEnricher, classify_content, classify_audience, compute_complexity, extract_entities


class TestClassifyContent:
    def test_methodology_detection(self):
        assert classify_content("This method uses a novel approach and technique") == "methodology"

    def test_results_detection(self):
        assert classify_content("Results show 95% accuracy in the benchmark evaluation") == "results"

    def test_attack_detection(self):
        assert classify_content("The injection attack exploits a vulnerability") == "attack"

    def test_defense_detection(self):
        assert classify_content("The defense provides protection and security mitigation") == "defense"

    def test_background_detection(self):
        assert classify_content("This introduction provides an overview of related work") == "background"

    def test_practical_detection(self):
        assert classify_content("How to follow this step-by-step guide with examples") == "practical"

    def test_other_fallback(self):
        assert classify_content("Random unrelated text here") == "other"


class TestClassifyAudience:
    def test_technical(self):
        assert classify_audience("The API deployment architecture has low latency") == "technical"

    def test_academic(self):
        assert classify_audience("Smith et al conducted an experiment in Table 1") == "academic"

    def test_general(self):
        assert classify_audience("Basically, imagine a simple easy solution") == "general"


class TestComputeComplexity:
    def test_beginner(self):
        assert compute_complexity("This is a simple easy text") == "beginner"

    def test_expert_with_formulas(self):
        assert compute_complexity("x = y + z where \\frac{a}{b} gives 0.95 score of 0.001") == "expert"


class TestExtractEntities:
    def test_extracts_capitalized(self):
        entities = extract_entities("Machine Learning and Natural Language Processing use CNN models")
        assert len(entities) > 0

    def test_extracts_acronyms(self):
        entities = extract_entities("The ASV and CNN metrics show LLM performance")
        assert any(e in ["ASV", "CNN", "LLM"] for e in entities)

    def test_max_five(self):
        entities = extract_entities("AAA BBB CCC DDD EEE FFF GGG HHH III JJJ")
        assert len(entities) <= 5


class TestProductionEnricher:
    def test_enriches_all_fields(self):
        enricher = ProductionEnricher()
        doc = Document(page_content="This method uses a CNN approach with 95.2% accuracy on the benchmark.", metadata={"page": 1})
        result = enricher.enrich(doc)
        assert "content_type" in result.metadata
        assert "audience" in result.metadata
        assert "complexity" in result.metadata
        assert "key_entities" in result.metadata
        assert "has_numbers" in result.metadata
        assert "has_table" in result.metadata
        assert "has_code" in result.metadata
        assert "word_count" in result.metadata

    def test_has_numbers_detection(self):
        enricher = ProductionEnricher()
        doc = Document(page_content="Results show 95.2% accuracy and 0.001 error rate.", metadata={})
        result = enricher.enrich(doc)
        assert result.metadata["has_numbers"] is True

    def test_has_code_detection(self):
        enricher = ProductionEnricher()
        doc = Document(page_content="```python\ndef hello(): pass\n```", metadata={})
        result = enricher.enrich(doc)
        assert result.metadata["has_code"] is True

    def test_batch_enrichment(self):
        enricher = ProductionEnricher()
        docs = [
            Document(page_content="Method A uses technique B", metadata={"page": 1}),
            Document(page_content="Attack injection vulnerability exploit", metadata={"page": 2}),
        ]
        results = enricher.enrich_batch(docs)
        assert len(results) == 2
        assert all("content_type" in r.metadata for r in results)