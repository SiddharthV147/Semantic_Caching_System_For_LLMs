import json
import logging
import uuid
from dataclasses import dataclass
from typing import Optional

from pymilvus import MilvusException

from config.constants import (
    CACHE_COLLECTION_NAME,
    CACHE_SEARCH_PARAMS,
    FIELD_QUERY_VECTOR,
    FIELD_COURSE_TAG,
    FIELD_REDIS_KEY,
    FIELD_PK,
    REDIS_CACHE_PREFIX,
    TOP_K_CACHE,
)
from config.settings import settings
from src.cache.eviction import LFUEvictionPolicy
from src.database.milvus_client import get_milvus_client
from src.database.redis_client import get_redis_client
from src.embeddings.embedding_service import get_embedder

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CacheHit:
    redis_key:  str
    course_tag: str
    response:   str
    similarity: float
    query_text: str

    @property
    def is_hit(self) -> bool:
        return True


@dataclass(frozen=True)
class CacheMiss:
    query_text: str
    course_tag: str
    best_score: float

    @property
    def is_hit(self) -> bool:
        return False


CacheResult = CacheHit | CacheMiss


# ─────────────────────────────────────────────────────────────────────────────
# Cache Manager
# ─────────────────────────────────────────────────────────────────────────────

class CacheManager:

    def __init__(self) -> None:
        self._milvus   = get_milvus_client()
        self._redis    = get_redis_client()
        self._embedder = get_embedder()
        self._eviction = LFUEvictionPolicy(
            redis=self._redis,
            milvus=self._milvus,
        )

    # ── Read path ─────────────────────────────────────────────────────────────

    def search_cache(
        self,
        query_text:           str,
        course_tag:           str,
        similarity_threshold: Optional[float] = None,
    ) -> CacheResult:
        """
        Search for a semantically similar cached answer, scoped to course_tag.
        On a HIT, increments the LFU hit counter and recomputes the score.
        """
        threshold    = similarity_threshold or settings.similarity_threshold
        query_vector = self._embedder.embed(query_text)
        expr         = f'{FIELD_COURSE_TAG} == "{course_tag}"'

        try:
            results = self._milvus.search(
                collection_name=CACHE_COLLECTION_NAME,
                data=[query_vector],
                anns_field=FIELD_QUERY_VECTOR,
                search_params=CACHE_SEARCH_PARAMS,
                limit=TOP_K_CACHE,
                filter=expr,
                output_fields=[FIELD_REDIS_KEY, FIELD_COURSE_TAG],
                consistency_level="Strong",
            )
        except MilvusException as exc:
            logger.error("Milvus search failed | course=%s | %s", course_tag, exc)
            raise

        hits = results[0] if results else []
        if not hits:
            return CacheMiss(query_text=query_text, course_tag=course_tag, best_score=0.0)

        top_hit   = hits[0]
        score     = top_hit.get("distance", 0.0)
        redis_key = top_hit["entity"][FIELD_REDIS_KEY]

        if score < threshold:
            return CacheMiss(query_text=query_text, course_tag=course_tag, best_score=score)

        raw = self._redis.get(redis_key)
        if raw is None:
            logger.warning("Stale entry: Milvus hit but Redis key '%s' missing.", redis_key)
            return CacheMiss(query_text=query_text, course_tag=course_tag, best_score=score)

        payload  = json.loads(raw)
        response = payload.get("response", "")

        # ── LFU: record this hit ──────────────────────────────────────────────
        self._eviction.record_hit(redis_key=redis_key, course_tag=course_tag)

        logger.info("Cache HIT | course=%s | score=%.4f | key=%s", course_tag, score, redis_key)
        return CacheHit(
            redis_key=redis_key,
            course_tag=course_tag,
            response=response,
            similarity=score,
            query_text=query_text,
        )

    # ── Write path ────────────────────────────────────────────────────────────

    def update_cache(
        self,
        query_text:    str,
        response_text: str,
        course_tag:    str,
        ttl_seconds:   Optional[int] = None,
    ) -> str:
        """
        Store a new query→response pair in the semantic cache.

        Write order:
          1. Evict if the course cache is at capacity (before adding new entry)
          2. Write payload to Redis
          3. Insert vector into Milvus (captures pk)
          4. Register entry with LFU tracker (stores pk + adds to ZSET)

        On any failure, rolls back in reverse order to keep stores consistent.
        """
        import time as _time

        ttl       = ttl_seconds or settings.cache_ttl_seconds
        redis_key = f"{REDIS_CACHE_PREFIX}:{course_tag}:{uuid.uuid4().hex}"

        # ── Step 1: Evict if needed ───────────────────────────────────────────
        try:
            evicted = self._eviction.evict_if_needed(course_tag=course_tag)
            if evicted:
                logger.info("Pre-insert eviction: removed %d entries for course '%s'.", evicted, course_tag)
        except Exception as exc:
            # Eviction failure is non-fatal for the write — log and continue
            logger.warning("Eviction check failed (non-fatal): %s", exc)

        # ── Step 2: Embed ─────────────────────────────────────────────────────
        query_vector = self._embedder.embed(query_text)

        # ── Step 3: Write Redis payload (before Milvus so we can roll back) ───
        payload = json.dumps({
            "response":          response_text,
            "query_text":        query_text,
            "course_tag":        course_tag,
            "hit_count":         0,           # will be incremented by record_hit()
            "created_at":        _time.time(),
            "last_accessed_at":  _time.time(),
        })
        self._redis.setex(name=redis_key, time=ttl, value=payload)

        # ── Step 4: Insert vector into Milvus ─────────────────────────────────
        try:
            result = self._milvus.insert(
                collection_name=CACHE_COLLECTION_NAME,
                data=[{
                    FIELD_QUERY_VECTOR: query_vector,
                    FIELD_COURSE_TAG:   course_tag,
                    FIELD_REDIS_KEY:    redis_key,
                }],
            )
            self._milvus.flush(collection_name=CACHE_COLLECTION_NAME)

            # Extract the auto-generated primary key from the insert result
            milvus_pk = result.primary_keys[0]

        except MilvusException as exc:
            # Roll back Redis write
            self._redis.delete(redis_key)
            logger.error(
                "Milvus insert failed — Redis key '%s' rolled back. Error: %s",
                redis_key, exc,
            )
            raise

        # ── Step 5: Register with LFU tracker ────────────────────────────────
        try:
            self._eviction.register_entry(
                redis_key=redis_key,
                course_tag=course_tag,
                milvus_pk=milvus_pk,
                ttl=ttl,
            )
        except Exception as exc:
            # LFU registration failure doesn't break the cache — entry is usable
            # but won't participate in eviction decisions until it gets a hit
            logger.warning("LFU registration failed for '%s' (non-fatal): %s", redis_key, exc)

        logger.info(
            "Cache WRITE | course=%s | key=%s | pk=%d",
            course_tag, redis_key, milvus_pk,
        )
        return redis_key

    # ── Utilities ─────────────────────────────────────────────────────────────

    def invalidate(self, redis_key: str, course_tag: str) -> None:
        """Remove a specific cache entry (called from the invalidation API)."""
        self._eviction._evict_one(
            redis_key=redis_key,
            course_tag=course_tag,
            score=0.0,
        )

    def get_eviction_stats(self, course_tag: str) -> dict:
        """Return LFU statistics for a course — used by the admin API."""
        return self._eviction.get_course_stats(course_tag)


# ── Module-level singleton ────────────────────────────────────────────────────
_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    global _manager
    if _manager is None:
        _manager = CacheManager()
    return _manager
