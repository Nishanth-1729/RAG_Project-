"""Central configuration. Everything is overridable through environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _p(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else ROOT / p


def _model(env: str, default: str) -> str:
    """HF model id, or a local checkpoint path (relative paths are resolved against the project root)."""
    v = os.getenv(env, default)
    return str(ROOT / v) if not Path(v).is_absolute() and (ROOT / v).exists() else v


def _db_url() -> str:
    url = os.getenv("DATABASE_URL", "sqlite:///storage/scholar.db")
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        rel = url[len("sqlite:///"):]
        (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{ROOT / rel}"
    return url


@dataclass
class Settings:
    # paths
    root: Path = ROOT
    data_dir: Path = field(default_factory=lambda: _p("DATA_DIR", "data"))
    raw_dir: Path = field(default_factory=lambda: _p("RAW_DIR", "data/raw_papers"))
    upload_dir: Path = field(default_factory=lambda: _p("UPLOAD_DIR", "data/uploads"))
    parsed_dir: Path = field(default_factory=lambda: _p("PARSED_DIR", "data/parsed"))
    eval_dir: Path = field(default_factory=lambda: _p("EVAL_DIR", "data/evaluation"))
    cache_dir: Path = field(default_factory=lambda: _p("CACHE_DIR", "data/cache"))
    results_dir: Path = field(default_factory=lambda: _p("RESULTS_DIR", "results"))
    models_dir: Path = field(default_factory=lambda: _p("MODELS_DIR", "models"))
    log_dir: Path = field(default_factory=lambda: _p("LOG_DIR", "logs"))

    # LLM (OpenAI-compatible)
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "EMPTY")
    llm_model: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    llm_max_context: int = int(os.getenv("LLM_MAX_CONTEXT", "12288"))
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "180"))
    # price per 1M tokens (USD) for cost accounting; local model = 0
    llm_price_in: float = float(os.getenv("LLM_PRICE_IN", "0"))
    llm_price_out: float = float(os.getenv("LLM_PRICE_OUT", "0"))

    # retrieval models
    embed_model: str = _model("EMBED_MODEL", "BAAI/bge-m3")
    rerank_model: str = _model("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    device: str = os.getenv("DEVICE", "cuda")
    fusion: str = os.getenv("FUSION", "rsf")                      # rsf (relative score fusion) | rrf
    fusion_weights: str = os.getenv("FUSION_WEIGHTS", "0.4,0.6")   # (dense, bm25) - chosen on the DEV split
    embed_batch: int = int(os.getenv("EMBED_BATCH", "16"))   # small: the LLM server shares the 16 GB card
    rerank_batch: int = int(os.getenv("RERANK_BATCH", "8"))

    # stores
    qdrant_url: str = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "scholar_chunks")
    database_url: str = field(default_factory=_db_url)

    # academic APIs
    s2_api_key: str | None = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
    openalex_api_key: str | None = os.getenv("OPENALEX_API_KEY") or None
    contact_email: str | None = os.getenv("CONTACT_EMAIL") or None

    # chunking defaults
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "section")
    chunk_tokens: int = int(os.getenv("CHUNK_TOKENS", "384"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "48"))

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.raw_dir, self.upload_dir, self.parsed_dir, self.eval_dir,
                  self.cache_dir, self.results_dir, self.models_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
