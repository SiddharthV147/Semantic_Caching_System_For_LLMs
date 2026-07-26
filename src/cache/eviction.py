import json
import logging
import time
from typing import Optional

from redis import Redis
from pymilvus import MilvusClient, MilvusException

from config.constants import (
    CACHE_COLLECTION_NAME,
    REDIS_CACHE_PREFIX,
    REDIS_FREQ_ZSET_PREFIX,
    REDIS_PK_MAP_PREFIX,
)
from config.settings import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LFU score computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_lfu_score(
    hit_count:  int,
    created_at: float,
    decay_rate: Optional[float] = None,
) -> float:
    """
    Compute the LFU eviction score for a cache entry.

    Score = hit_count / (1 + age_hours * decay_rate)

    Args:
        hit_count:   Number of cache hits for this entry.
        created_at:  Unix timestamp (seconds) when the entry was created.
        decay_rate:  Decay coefficient. Defaults to settings.lfu_decay_rate.

    Returns:
        Float eviction score. Higher = more valuable = kept longer.
    """
    rate      = decay_rate if decay_rate is not None else settings.lfu_decay_rate
    now       = time.time()
    age_hours = max(0.0, (now - created_at) / 3600.0)
    score     = hit_count / (1.0 + age_hours * rate)
    return round(score, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Eviction engine
# ─────────────────────────────────────────────────────────────────────────────

class LFUEvictionPolicy:
    """
    LFU eviction engine backed entirely by Redis — no in-process state.

    Multiple API workers can use this safely because every operation
    goes through Redis, which serialises concurrent writes atomically
    via its single-threaded command processing.

    Integration points in CacheManager
    ------------------------------------
    On write (update_cache):
        eviction.evict_if_needed(course_tag)      # before insert
        eviction.register_entry(...)               # after successful insert

    On read hit (search_cache):
        eviction.record_hit(redis_key, course_tag) # after Redis GET
    """

    def __init__(self, redis: Redis, milvus: MilvusClient) -> None:
        self._redis  = redis
        self._milvus = milvus

    # ── Key helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _freq_key(course_tag: str) -> str:
        return f"{REDIS_FREQ_ZSET_PREFIX}:{course_tag}"

    @staticmethod
    def _pk_key(redis_key: str) -> str:
        return f"{REDIS_PK_MAP_PREFIX}:{redis_key}"

    # ── Register new entry ────────────────────────────────────────────────────

    def register_entry(
        self,
        redis_key:  str,
        course_tag: str,
        milvus_pk:  int,
        ttl:        int,
    ) -> None:
        """
        Track a newly inserted cache entry in the LFU system.

        Called after Milvus insert + Redis payload write succeed.

        Stores the Milvus PK so we can delete the vector later during eviction.
        Adds the entry to the course's frequency ZSET with score=1.0 (new entry).
        """
        # Store Milvus PK with same TTL as payload — auto-expires in sync
        self._redis.setex(
            name=self._pk_key(redis_key),
            time=ttl,
            value=str(milvus_pk),
        )

        # Score for a brand-new entry (hit_count=1, age=0)
        initial_score = compute_lfu_score(hit_count=1, created_at=time.time())
        self._redis.zadd(self._freq_key(course_tag), {redis_key: initial_score})

        logger.debug(
            "LFU registered | course=%s | key=%s | pk=%d | score=%.4f",
            course_tag, redis_key, milvus_pk, initial_score,
        )

    # ── Record hit ────────────────────────────────────────────────────────────

    def record_hit(self, redis_key: str, course_tag: str) -> None:
        """
        Increment hit count and recompute the LFU score after a cache hit.

        We recompute the FULL score (not just ZINCRBY) so that age decay is
        always applied correctly — ZINCRBY would ignore how old the entry is.

        Called by CacheManager.search_cache() on every HIT.
        """
        payload_raw = self._redis.get(redis_key)
        if payload_raw is None:
            # Entry expired from Redis but is still in ZSET — clean up ghost
            self._redis.zrem(self._freq_key(course_tag), redis_key)
            logger.warning(
                "LFU ghost entry removed from ZSET | key='%s' expired from Redis.", redis_key
            )
            return

        try:
            payload             = json.loads(payload_raw)
            hit_count           = payload.get("hit_count", 0) + 1
            created_at          = payload.get("created_at", time.time())
            payload["hit_count"]          = hit_count
            payload["last_accessed_at"]   = time.time()

            # Preserve remaining TTL — hitting a cache entry does not reset its lifetime
            remaining_ttl = self._redis.ttl(redis_key)
            if remaining_ttl > 0:
                self._redis.setex(redis_key, remaining_ttl, json.dumps(payload))

            # Recompute and update ZSET score
            new_score = compute_lfu_score(hit_count=hit_count, created_at=created_at)
            self._redis.zadd(self._freq_key(course_tag), {redis_key: new_score})

            logger.debug(
                "LFU hit | course=%s | key=%s | hits=%d | score=%.4f",
                course_tag, redis_key, hit_count, new_score,
            )

        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("LFU record_hit payload parse failed for '%s': %s", redis_key, exc)

    # ── Eviction decision ─────────────────────────────────────────────────────

    def evict_if_needed(self, course_tag: str) -> int:
        """
        Evict the lowest-scored entries if the course cache is at capacity.

        Called BEFORE inserting a new entry — guarantees we never exceed the cap.

        We evict in a batch (settings.cache_eviction_batch) to amortise overhead:
        instead of evicting 1 entry per insert when at capacity, we evict `batch`
        at once, making the next `batch-1` inserts free of eviction cost.

        Returns the number of entries evicted (0 if no eviction was needed).
        """
        freq_key      = self._freq_key(course_tag)
        current_count = self._redis.zcard(freq_key)
        max_entries   = settings.cache_max_entries_per_course

        if current_count < max_entries:
            return 0

        batch      = settings.cache_eviction_batch
        candidates = self._redis.zrange(freq_key, 0, batch - 1, withscores=True)

        logger.info(
            "LFU eviction | course=%s | count=%d/%d | evicting %d entries",
            course_tag, current_count, max_entries, len(candidates),
        )

        evicted = 0
        for redis_key, score in candidates:
            self._evict_one(redis_key=redis_key, course_tag=course_tag, score=score)
            evicted += 1

        return evicted

    # ── Evict one entry ───────────────────────────────────────────────────────

    def _evict_one(self, redis_key: str, course_tag: str, score: float) -> None:
        """
        Remove one cache entry from Milvus + Redis atomically (best-effort).

        Deletion order is intentional:
          1. Milvus first — if this fails, keep Redis intact (data stays consistent)
          2. Redis payload
          3. ZSET membership
          4. PK mapping

        If Milvus fails, we abort and leave the entry in place.
        It will appear in the next eviction batch and we'll retry.
        """
        pk_raw = self._redis.get(self._pk_key(redis_key))

        # Step 1 — delete vector from Milvus
        if pk_raw:
            try:
                milvus_pk = int(pk_raw)
                self._milvus.delete(
                    collection_name=CACHE_COLLECTION_NAME,
                    ids=[milvus_pk],
                )
            except (MilvusException, ValueError) as exc:
                logger.error(
                    "Milvus delete failed (pk=%s) — aborting eviction of '%s': %s",
                    pk_raw, redis_key, exc,
                )
                return  # Do NOT delete Redis data if Milvus failed
        else:
            logger.debug(
                "No Milvus PK for '%s' — vector may already be gone, cleaning Redis.",
                redis_key,
            )

        # Steps 2-4 — clean up Redis
        pipeline = self._redis.pipeline(transaction=True)
        pipeline.delete(redis_key)                          # payload
        pipeline.zrem(self._freq_key(course_tag), redis_key)  # ZSET
        if pk_raw:
            pipeline.delete(self._pk_key(redis_key))        # PK map
        pipeline.execute()

        logger.info(
            "LFU evicted | course=%s | key=%s | score=%.4f",
            course_tag, redis_key, score,
        )

    # ── Admin utilities ───────────────────────────────────────────────────────

    def get_course_stats(self, course_tag: str) -> dict:
        """
        Return LFU statistics for a course.
        Used by the health/admin API endpoint.
        """
        freq_key = self._freq_key(course_tag)
        count    = self._redis.zcard(freq_key)
        max_e    = settings.cache_max_entries_per_course

        top    = self._redis.zrange(freq_key, -5, -1, withscores=True, rev=True)
        bottom = self._redis.zrange(freq_key,  0,  4, withscores=True)

        return {
            "course_tag":      course_tag,
            "entry_count":     count,
            "max_entries":     max_e,
            "utilisation_pct": round(count / max_e * 100, 1) if max_e > 0 else 0.0,
            "top_entries":     [{"key": k, "score": round(s, 4)} for k, s in top],
            "bottom_entries":  [{"key": k, "score": round(s, 4)} for k, s in bottom],
        }

    def clear_course(self, course_tag: str) -> int:
        """
        Remove ALL cache entries for a course — used by cache invalidation API.
        Returns number of entries removed.
        """
        freq_key    = self._freq_key(course_tag)
        all_entries = self._redis.zrange(freq_key, 0, -1, withscores=True)
        count       = 0

        for redis_key, score in all_entries:
            self._evict_one(redis_key=redis_key, course_tag=course_tag, score=score)
            count += 1

        self._redis.delete(freq_key)
        logger.info("LFU cleared %d entries for course '%s'.", count, course_tag)
        return count
