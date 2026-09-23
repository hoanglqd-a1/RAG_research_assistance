"""Environment-based settings for the retrieval pipeline."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, read from environment variables."""

    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "75"))
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))


settings = Settings()
