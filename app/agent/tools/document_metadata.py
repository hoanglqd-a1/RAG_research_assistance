"""Indexed-document catalog exposed as an agent tool."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent.registry import Tool
from app.rag.vector_store import VectorStore


class GetDocumentMetadataInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str | None = Field(
        default=None,
        description="Exact filename to inspect; omit to list every indexed document",
    )


def create_document_metadata_tool(
    vector_store: VectorStore,
) -> Tool[GetDocumentMetadataInput]:
    def get_document_metadata(
        arguments: GetDocumentMetadataInput,
    ) -> dict[str, object]:
        chunks = vector_store.list_chunks(arguments.filename)
        grouped: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            entry = grouped.setdefault(
                chunk.filename,
                {
                    "filename": chunk.filename,
                    "chunk_count": 0,
                    "pages": set(),
                    "metadata": {},
                },
            )
            entry["chunk_count"] += 1
            if chunk.page is not None:
                entry["pages"].add(chunk.page)
            entry["metadata"].update(chunk.metadata)

        documents = []
        for filename in sorted(grouped):
            entry = grouped[filename]
            documents.append(
                {
                    "filename": entry["filename"],
                    "chunk_count": entry["chunk_count"],
                    "pages": sorted(entry["pages"]),
                    "metadata": entry["metadata"],
                }
            )
        return {
            "documents": documents,
            "count": len(documents),
            "requested_filename": arguments.filename,
        }

    return Tool(
        name="get_document_metadata",
        description=(
            "List indexed documents or inspect one exact filename, including its "
            "page numbers, chunk count, and stored metadata."
        ),
        input_model=GetDocumentMetadataInput,
        handler=get_document_metadata,
    )
