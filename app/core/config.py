"""Environment-based application settings."""

from dataclasses import dataclass
import math
import os

from dotenv import load_dotenv


# Loading here keeps scripts and Uvicorn consistent. Existing process environment
# variables still take precedence over values in the local .env file.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, read from environment variables."""

    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "75"))
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "dense")
    bm25_k1: float = float(os.getenv("BM25_K1", "1.5"))
    bm25_b: float = float(os.getenv("BM25_B", "0.75"))
    hybrid_candidate_k: int = int(os.getenv("HYBRID_CANDIDATE_K", "20"))
    hybrid_rrf_k: float = float(os.getenv("HYBRID_RRF_K", "60"))
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "qwen3:14b-q4_K_M")
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "6"))
    summary_batch_chars: int = int(os.getenv("SUMMARY_BATCH_CHARS", "8000"))
    max_summary_batches: int = int(os.getenv("MAX_SUMMARY_BATCHES", "12"))

    def __post_init__(self) -> None:
        if self.retrieval_mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError("RETRIEVAL_MODE must be dense, bm25, or hybrid")
        if self.hybrid_candidate_k <= 0:
            raise ValueError("HYBRID_CANDIDATE_K must be greater than zero")
        if not math.isfinite(self.hybrid_rrf_k) or self.hybrid_rrf_k <= 0:
            raise ValueError("HYBRID_RRF_K must be finite and greater than zero")


settings = Settings()
