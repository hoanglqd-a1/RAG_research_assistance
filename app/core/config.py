"""Environment-based application settings."""

from dataclasses import dataclass
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
    llm_api_key: str | None = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "6"))
    summary_batch_chars: int = int(os.getenv("SUMMARY_BATCH_CHARS", "8000"))
    max_summary_batches: int = int(os.getenv("MAX_SUMMARY_BATCHES", "12"))


settings = Settings()
