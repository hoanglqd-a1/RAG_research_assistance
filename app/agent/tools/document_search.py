"""Document retrieval exposed as a validated agent tool."""

from pydantic import BaseModel, ConfigDict, Field

from app.agent.registry import Tool
from app.rag.retriever import RetrievalProvider


class SearchDocumentsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Document search query")
    top_k: int = Field(default=5, ge=1, le=20)


def create_search_documents_tool(retriever: RetrievalProvider) -> Tool[SearchDocumentsInput]:
    """Use the configured retrieval backend."""

    def search_documents(arguments: SearchDocumentsInput) -> dict[str, object]:
        results = retriever.retrieve(arguments.query, arguments.top_k)
        return {
            "query": arguments.query,
            "results": [
                {
                    "text": result.chunk.text,
                    "filename": result.chunk.filename,
                    "page": result.chunk.page,
                    "chunk_id": result.chunk.chunk_id,
                    "score": result.score,
                    "metadata": result.chunk.metadata,
                }
                for result in results
            ],
        }

    return Tool(
        name="search_documents",
        description=(
            "Search indexed documents for passages relevant to a question. "
            "Returns text, source metadata, and backend-specific relevance scores."
        ),
        input_model=SearchDocumentsInput,
        handler=search_documents,
    )
