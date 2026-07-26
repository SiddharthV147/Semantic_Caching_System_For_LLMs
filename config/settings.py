"""
config/settings.py
Centralised configuration using Pydantic BaseSettings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):

    # ── Milvus ───────────────────────────────────────────────────────────────
    milvus_uri: str = Field(default="http://localhost:19530")
    milvus_token: str = Field(default="")
    milvus_db_name: str = Field(default="default")

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_max_connections: int = Field(default=20)
    redis_socket_timeout: float = Field(default=2.0)

    # ── Embedding model ───────────────────────────────────────────────────────
    embedding_model_name: str = Field(default="BAAI/bge-large-en-v1.5")
    embedding_dim: int = Field(default=1024)
    embedding_device: str = Field(default="cpu")
    embedding_batch_size: int = Field(default=32)
    embedding_normalize: bool = Field(default=True)

    # ── LLM — HuggingFace Inference Providers ────────────────────────────────
    hf_token: str = Field(default="")
    llm_model_name: str = Field(default="Qwen/Qwen2.5-7B-Instruct")
    llm_provider: str = Field(default="auto")
    llm_max_new_tokens: int = Field(default=512)
    llm_temperature: float = Field(default=0.3)
    llm_backend: str = Field(default="huggingface_api")
    llm_timeout: int = Field(default=60)

    # ── Cache behaviour ───────────────────────────────────────────────────────
    similarity_threshold: float = Field(default=0.92)
    cache_ttl_seconds: int = Field(default=86_400)

    # ── Eviction (LFU with decay) ─────────────────────────────────────────────
    cache_max_entries_per_course: int = Field(
        default=1000,
        description=(
            "Maximum cache entries stored per course before eviction triggers. "
            "When the course cache is full, the entry with the lowest LFU score "
            "(least frequently used, adjusted for age) is evicted."
        ),
    )
    cache_eviction_batch: int = Field(
        default=10,
        description=(
            "Number of entries to evict in a single batch when the cache is full. "
            "Batching reduces the frequency of eviction overhead. "
            "E.g. 10 means: evict 10 at once when count > max, "
            "then the next eviction won't happen until 10 new entries are added."
        ),
    )
    lfu_decay_rate: float = Field(
        default=0.01,
        description=(
            "LFU aging decay rate. Score = hit_count / (1 + age_hours * decay_rate). "
            "0.01 means frequency halves every ~70 hours. "
            "Prevents old high-frequency entries from permanently blocking new ones."
        ),
    )

    # ── API server ────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=1)
    api_reload: bool = Field(default=False)
    api_title: str = Field(default="LMS Semantic Cache API")
    api_version: str = Field(default="1.0.0")
    log_level: str = Field(default="info")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()