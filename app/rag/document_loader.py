"""Load PDF and plain-text files without hiding extraction behavior."""

import logging
from pathlib import Path
import re

import pymupdf

from app.rag.models import Document

logger = logging.getLogger(__name__)


class UnsupportedDocumentError(ValueError):
    """Raised when a file type is not supported by this first version."""


def clean_text(text: str) -> str:
    """Remove common extraction noise while preserving paragraph boundaries."""

    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class DocumentLoader:
    """Extract clean text and source metadata from supported files."""

    def load(self, path: str | Path) -> list[Document]:
        """Return one Document per PDF page, or one for a text file."""

        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Document does not exist: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(file_path)
        if suffix in {".txt", ".text"}:
            return self._load_text(file_path)
        raise UnsupportedDocumentError(
            f"Unsupported file type '{suffix}'. Only PDF and text files are supported."
        )

    @staticmethod
    def _load_text(path: Path) -> list[Document]:
        text = clean_text(path.read_text(encoding="utf-8-sig"))
        logger.info("Loaded text document %s", path.name)
        return [Document(text=text, filename=path.name)] if text else []

    @staticmethod
    def _load_pdf(path: Path) -> list[Document]:
        documents: list[Document] = []
        try:
            with pymupdf.open(path) as pdf:
                for page_index, page in enumerate(pdf):
                    text = clean_text(page.get_text("text"))
                    if text:
                        documents.append(
                            Document(
                                text=text,
                                filename=path.name,
                                page=page_index + 1,
                            )
                        )
        except pymupdf.FileDataError as exc:
            raise ValueError(f"Could not read PDF '{path.name}': {exc}") from exc

        logger.info("Loaded %d non-empty pages from %s", len(documents), path.name)
        return documents
