import logging
import threading
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output vector dimension."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single string. Returns a plain Python list of floats."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in one forward pass."""

class HuggingFaceEmbedder(BaseEmbedder):

    _BGE_QUERY_PREFIX = "Represent this sentence: "

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer   # lazy import

        model_name = settings.embedding_model_name
        device     = settings.embedding_device

        logger.info("Loading HuggingFace model '%s' on device='%s' …", model_name, device)
        self._model      = SentenceTransformer(model_name, device=device)
        self._dim        = settings.embedding_dim
        self._normalize  = settings.embedding_normalize
        self._batch_size = settings.embedding_batch_size
        self._use_prefix = "bge" in model_name.lower()

        # Sanity-check: actual model dim must match config
        probe = self._model.encode("probe", normalize_embeddings=False)
        actual_dim = len(probe)
        if actual_dim != self._dim:
            raise ValueError(
                f"Model '{model_name}' produces {actual_dim}-dim vectors, "
                f"but settings.embedding_dim={self._dim}. "
                "Update settings.py or switch models."
            )
        logger.info("Model loaded. dim=%d  normalize=%s", self._dim, self._normalize)

    @property
    def dim(self) -> int:
        return self._dim

    def _apply_prefix(self, text: str) -> str:
        return f"{self._BGE_QUERY_PREFIX}{text}" if self._use_prefix else text

    def embed(self, text: str) -> list[float]:
        prepared = self._apply_prefix(text)
        vector = self._model.encode(
            prepared,
            normalize_embeddings=self._normalize,
            batch_size=1,
        )
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        prepared = [self._apply_prefix(t) for t in texts]
        vectors = self._model.encode(
            prepared,
            normalize_embeddings=self._normalize,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return vectors.tolist()


class MockEmbedder(BaseEmbedder):

    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim
        logger.warning("MockEmbedder active — NOT suitable for production.")

    @property
    def dim(self) -> int:
        return self._dim

    def _make_vector(self, text: str) -> list[float]:
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**32))
        vec = rng.standard_normal(self._dim).astype(np.float32)
        vec /= np.linalg.norm(vec)          # L2 normalise
        return vec.tolist()

    def embed(self, text: str) -> list[float]:
        return self._make_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._make_vector(t) for t in texts]


class EmbedderFactory:
    """Thread-safe singleton factory. Call get_embedder() everywhere."""

    _instance: Optional[BaseEmbedder] = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def _build(cls) -> BaseEmbedder:
        backend = getattr(settings, "embedding_backend", "huggingface").lower()
        if backend == "mock":
            return MockEmbedder(dim=settings.embedding_dim)
        return HuggingFaceEmbedder()

    @classmethod
    def get(cls) -> BaseEmbedder:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls._build()
        return cls._instance


def get_embedder() -> BaseEmbedder:
    """Module-level accessor. Import and call this everywhere."""
    return EmbedderFactory.get()