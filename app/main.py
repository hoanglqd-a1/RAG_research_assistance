"""Local web API and static UI. Run with uvicorn app.main:app."""

from copy import deepcopy
from dataclasses import asdict
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 20 * 1024 * 1024


class Question(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = Field(default=None, max_length=100)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("message")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Câu hỏi không được để trống")
        return value.strip()


def create_app(services: Any = None) -> FastAPI:
    """Allow deterministic service injection without loading embedding models."""
    api = FastAPI(title="Paper Assistant", version="1.0.0")
    api.state.services = services
    lock = RLock()

    def get_services():
        if api.state.services is None:
            from app.services import build_services

            api.state.services = build_services(settings)
        return api.state.services

    def execute(operation):
        with lock:
            try:
                return operation(get_services())
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("Web operation failed")
                raise HTTPException(
                    503, "Không thể xử lý yêu cầu. Kiểm tra Ollama, model và log backend."
                ) from exc

    @api.get("/api/health")
    def health():
        return {"status": "ok", "model": settings.llm_model,
                "retrieval_mode": settings.retrieval_mode,
                "initialized": api.state.services is not None}

    @api.get("/api/documents")
    def documents():
        def listing(svc):
            grouped = {}
            for chunk in svc.vector_store.list_chunks():
                item = grouped.setdefault(chunk.filename, {"filename": chunk.filename,
                                                           "chunks": 0, "pages": set()})
                item["chunks"] += 1
                if chunk.page is not None:
                    item["pages"].add(chunk.page)
            return [{**item, "pages": len(item["pages"])} for item in grouped.values()]
        # Opening the UI should not trigger an embedding-model download.
        with lock:
            return [] if api.state.services is None else execute(listing)

    @api.post("/api/documents", status_code=201)
    def upload(file: UploadFile = File(...)):
        try:
            filename = (file.filename or "").replace("\\", "/").split("/")[-1]
            if not filename or Path(filename).suffix.lower() not in {".pdf", ".txt", ".text"}:
                raise HTTPException(415, "Chỉ hỗ trợ PDF và TXT (UTF-8).")
            with TemporaryDirectory(prefix="paper-upload-") as directory:
                path = Path(directory) / filename
                size = 0
                with path.open("wb") as target:
                    while block := file.file.read(1024 * 1024):
                        size += len(block)
                        if size > MAX_UPLOAD:
                            raise HTTPException(413, "Dung lượng tối đa là 20 MB mỗi file.")
                        target.write(block)
                if not size:
                    raise HTTPException(422, "File rỗng.")

                def ingest(svc):
                    if svc.vector_store.list_chunks(filename):
                        raise HTTPException(409, "Tên file đã tồn tại. Hãy đổi tên file trước khi tải lên.")
                    try:
                        chunks = svc.pipeline.ingest(path)
                    except ValueError as exc:
                        raise HTTPException(422, "File không hợp lệ hoặc không có văn bản đọc được.") from exc
                    return {"filename": filename, "chunks": len(chunks)}
                return execute(ingest)
        finally:
            file.file.close()

    def require_documents(svc):
        if not svc.vector_store.list_chunks():
            raise HTTPException(409, "Hãy tải ít nhất một paper trước khi truy vấn.")

    @api.post("/api/search")
    def search(question: Question):
        def retrieve(svc):
            require_documents(svc)
            return {"results": [asdict(result) for result in
                                svc.pipeline.query(question.message, question.top_k)]}
        return execute(retrieve)

    @api.post("/api/chat")
    def chat(question: Question):
        def answer(svc):
            require_documents(svc)
            identifier, state = svc.conversations.get_or_create(question.conversation_id)
            # Commit the turn only after a successful model call.
            working = deepcopy(state)
            result = svc.agent.run(question.message, state=working)
            state.events[:] = working.events
            return {"conversation_id": identifier, **result.model_dump()}
        return execute(answer)

    api.mount("/static", StaticFiles(directory=STATIC), name="static")

    @api.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    return api


app = create_app()
