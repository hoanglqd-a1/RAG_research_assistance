"""HTTP contract checks without Ollama or embedding downloads."""

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from app.agent.schemas import AgentRunResult
from app.agent.state import ConversationStore
from app.main import create_app
from app.rag.models import SearchResult, TextChunk


@pytest.fixture
def web():
    chunks = []

    def ingest(path):
        text = Path(path).read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("No text")
        chunk = TextChunk(text, Path(path).name, None, "chunk-1")
        chunks.append(chunk)
        return [chunk]

    def run(message, state):
        state.add_user_message(message)
        if message == "fail":
            raise RuntimeError("Ollama unavailable")
        state.add_assistant_message("Grounded answer")
        return AgentRunResult(answer="Grounded answer")

    services = SimpleNamespace(
        vector_store=SimpleNamespace(list_chunks=lambda filename=None: [
            chunk for chunk in chunks if filename is None or chunk.filename == filename]),
        pipeline=SimpleNamespace(ingest=ingest, query=lambda message, top_k: [
            SearchResult(chunk, 0.9) for chunk in chunks[:top_k]]),
        agent=SimpleNamespace(run=run), conversations=ConversationStore(),
    )
    with TestClient(create_app(services)) as client:
        yield client, services


def test_static_and_lazy_health():
    with TestClient(create_app()) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/documents").json() == []
        assert client.get("/api/health").json()["initialized"] is False


def test_upload_search_and_followup(web):
    client, services = web
    assert client.post("/api/chat", json={"message": "hello"}).status_code == 409
    response = client.post("/api/documents", files={"file": ("../paper.txt", b"Research evidence")})
    assert response.status_code == 201
    assert response.json() == {"filename": "paper.txt", "chunks": 1}
    assert client.get("/api/documents").json()[0]["chunks"] == 1
    result = client.post("/api/search", json={"message": "evidence", "top_k": 1}).json()
    assert result["results"][0]["chunk"]["text"] == "Research evidence"
    first = client.post("/api/chat", json={"message": "summarize"}).json()
    identifier = first["conversation_id"]
    second = client.post("/api/chat", json={"message": "explain", "conversation_id": identifier})
    assert second.json()["conversation_id"] == identifier
    _, state = services.conversations.get_or_create(identifier)
    assert len(state.events) == 4
    assert client.post("/api/chat", json={"message": "fail", "conversation_id": identifier}).status_code == 503
    assert len(state.events) == 4


def test_invalid_uploads_and_requests(web):
    client, _ = web
    assert client.post("/api/documents", files={"file": ("file.exe", b"bad")}).status_code == 415
    assert client.post("/api/documents", files={"file": ("empty.txt", b"")}).status_code == 422
    assert client.post("/api/documents", files={"file": ("blank.txt", b"   ")}).status_code == 422
    assert client.post("/api/documents", files={"file": ("large.txt", b"x" * (20 * 1024 * 1024 + 1))}).status_code == 413
    assert client.post("/api/documents", files={"file": ("paper.txt", b"text")}).status_code == 201
    assert client.post("/api/documents", files={"file": ("paper.txt", b"text")}).status_code == 409
    assert client.post("/api/chat", json={"message": "  "}).status_code == 422
    assert client.post("/api/search", json={"message": "test", "top_k": 0}).status_code == 422
