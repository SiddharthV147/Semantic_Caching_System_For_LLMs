import logging
import threading
from typing import Optional

import redis
from redis import Redis, ConnectionPool
from redis.exceptions import RedisError

from config.settings import settings

logger = logging.getLogger(__name__)


class RedisConnectionManager:

    _instance: Optional["RedisConnectionManager"] = None
    _lock: threading.Lock = threading.Lock()
    _pool: Optional[ConnectionPool] = None

    def __new__(cls) -> "RedisConnectionManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def _build_pool(self) -> ConnectionPool:
        logger.info("Building Redis connection pool → %s", settings.redis_url)
        return redis.ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_socket_timeout,
            decode_responses=True,   # all responses are str, not bytes
        )

    @property
    def pool(self) -> ConnectionPool:
        if self._pool is None:
            with self._lock:
                if self._pool is None:
                    self._pool = self._build_pool()
        return self._pool

    def get_client(self) -> Redis:
        """
        Return a Redis client backed by the shared pool.
        Callers should NOT close this client — the pool manages connections.
        """
        return Redis(connection_pool=self.pool)

    def ping(self) -> bool:
        """Health-check. Returns True if Redis is reachable."""
        try:
            return self.get_client().ping()
        except RedisError as exc:
            logger.warning("Redis ping failed: %s", exc)
            return False

    def close(self) -> None:
        """Disconnect all pool connections (call on app shutdown)."""
        if self._pool is not None:
            self._pool.disconnect()
            self._pool = None
            logger.info("Redis connection pool closed.")


# ── Module-level convenience accessor ────────────────────────────────────────
_manager = RedisConnectionManager()


def get_redis_client() -> Redis:
    """Return a Redis client from the shared pool. Thread-safe."""
    return _manager.get_client()