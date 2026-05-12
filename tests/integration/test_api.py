"""Integration tests — require Qdrant + API keys."""
import os, pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") or not os.getenv("VOYAGE_API_KEY"),
    reason="GEMINI_API_KEY and VOYAGE_API_KEY required",
)

@pytest.fixture(scope="module")
def client():
    from main import app
    return TestClient(app)

class TestHealth:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "healthy"
        assert "chunking_strategy" in d
        assert "crag_enabled" in d

class TestIngest:
    def test_reject_unsupported_format(self, client):
        assert client.post("/api/v1/ingest", files={"file": ("t.txt", b"hi", "text/plain")}).status_code == 400

    def test_ingest_with_enrichment(self, client, tmp_path):
        import fitz
        p = tmp_path / "test.pdf"
        doc = fitz.open(); pg = doc.new_page()
        pg.insert_text((72, 72), "Machine learning methodology involves training CNNs on large datasets. Results show 95% accuracy.")
        doc.save(str(p)); doc.close()
        with open(p, "rb") as f:
            r = client.post("/api/v1/ingest?enrich=true", files={"file": ("test.pdf", f, "application/pdf")})
        assert r.status_code == 200
        assert r.json()["metadata_enriched"] is True

class TestChat:
    def test_chat_with_verdict(self, client):
        r = client.post("/api/v1/chat", json={"query": "What methodology is used?"})
        if r.status_code == 200:
            d = r.json()
            assert "verdict" in d
            assert d["verdict"] in ("CORRECT", "AMBIGUOUS", "INCORRECT")

    def test_filtered_chat(self, client):
        r = client.post("/api/v1/chat", json={"query": "What are the results?", "content_type_filter": "results"})
        assert r.status_code in (200, 503)

    def test_validation(self, client):
        assert client.post("/api/v1/chat", json={"query": "ab"}).status_code == 422