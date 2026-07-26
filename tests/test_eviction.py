"""
tests/test_eviction.py

Tests for the LFU eviction engine and the updated CacheManager.

Design
------
- All Redis calls use fakeredis — a fully in-memory Redis implementation.
  No real Redis server needed.
- All Milvus calls are mocked via MagicMock.
- Tests are deterministic because compute_lfu_score uses time.time() which
  we freeze via unittest.mock.patch where needed.

Run
---
    pip install fakeredis
    pytest tests/test_eviction.py -v
    pytest tests/test_eviction.py::TestLFUScore -v
"""

import json
import time
import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import fakeredis

from src.cache.eviction import LFUEvictionPolicy, compute_lfu_score
from config.constants import (
    REDIS_FREQ_ZSET_PREFIX,
    REDIS_PK_MAP_PREFIX,
    REDIS_CACHE_PREFIX,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_redis():
    """In-memory Redis — no server needed."""
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def mock_milvus():
    m = MagicMock()
    m.delete.return_value = None
    return m


@pytest.fixture
def policy(fake_redis, mock_milvus):
    """LFUEvictionPolicy backed by fakeredis + mock Milvus."""
    return LFUEvictionPolicy(redis=fake_redis, milvus=mock_milvus)


def _seed_entry(
    fake_redis,
    policy: LFUEvictionPolicy,
    redis_key: str,
    course_tag: str,
    milvus_pk:  int,
    hit_count:  int = 1,
    created_at: float = None,
    ttl:        int = 86400,
):
    """Helper: insert a fully-formed cache entry into fakeredis + ZSET."""
    created_at = created_at or time.time()
    payload = json.dumps({
        "response":         f"Answer for {redis_key}",
        "query_text":       f"Query for {redis_key}",
        "course_tag":       course_tag,
        "hit_count":        hit_count,
        "created_at":       created_at,
        "last_accessed_at": created_at,
    })
    fake_redis.setex(redis_key, ttl, payload)

    score = compute_lfu_score(hit_count=hit_count, created_at=created_at)
    fake_redis.zadd(f"{REDIS_FREQ_ZSET_PREFIX}:{course_tag}", {redis_key: score})
    fake_redis.setex(f"{REDIS_PK_MAP_PREFIX}:{redis_key}", ttl, str(milvus_pk))
    return score


# ─────────────────────────────────────────────────────────────────────────────
# compute_lfu_score
# ─────────────────────────────────────────────────────────────────────────────

class TestLFUScore:

    def test_new_entry_score_is_one(self):
        """Brand-new entry: hit_count=1, age≈0 → score≈1.0"""
        score = compute_lfu_score(hit_count=1, created_at=time.time(), decay_rate=0.01)
        assert 0.99 < score <= 1.0

    def test_higher_hit_count_gives_higher_score(self):
        now = time.time()
        s1  = compute_lfu_score(hit_count=1,  created_at=now, decay_rate=0.01)
        s50 = compute_lfu_score(hit_count=50, created_at=now, decay_rate=0.01)
        assert s50 > s1

    def test_older_entry_scores_lower_than_same_frequency_recent(self):
        now    = time.time()
        recent = compute_lfu_score(hit_count=10, created_at=now - 3600,   decay_rate=0.01)
        old    = compute_lfu_score(hit_count=10, created_at=now - 360000, decay_rate=0.01)
        assert recent > old

    def test_decay_rate_zero_gives_pure_frequency(self):
        """With decay_rate=0, score = hit_count / 1.0 regardless of age."""
        ancient = time.time() - 10_000_000   # ~115 days ago
        score   = compute_lfu_score(hit_count=7, created_at=ancient, decay_rate=0.0)
        assert score == pytest.approx(7.0, abs=0.01)

    def test_high_decay_rate_strongly_penalises_old_entries(self):
        now   = time.time()
        score = compute_lfu_score(hit_count=100, created_at=now - 86400, decay_rate=1.0)
        # With rate=1.0 and age=24h: 100 / (1 + 24) = ~4.0
        assert score == pytest.approx(100 / 25, abs=0.5)

    def test_score_is_positive_for_any_valid_input(self):
        for hits in [1, 5, 100]:
            for age_secs in [0, 3600, 86400, 2_592_000]:
                score = compute_lfu_score(
                    hit_count=hits,
                    created_at=time.time() - age_secs,
                    decay_rate=0.01,
                )
                assert score > 0

    def test_eviction_ordering_is_correct(self):
        """
        Given four entries, verify the eviction order is correct:
        hot recent > hot old > cold recent > cold old
        """
        now = time.time()
        scores = {
            "hot_recent":  compute_lfu_score(50, now - 3600,   0.01),
            "hot_old":     compute_lfu_score(50, now - 360000, 0.01),
            "cold_recent": compute_lfu_score(1,  now - 3600,   0.01),
            "cold_old":    compute_lfu_score(1,  now - 360000, 0.01),
        }
        assert scores["hot_recent"] > scores["hot_old"]
        assert scores["hot_old"]    > scores["cold_recent"]
        assert scores["cold_recent"] > scores["cold_old"]


# ─────────────────────────────────────────────────────────────────────────────
# LFUEvictionPolicy.register_entry
# ─────────────────────────────────────────────────────────────────────────────

class TestRegisterEntry:

    def test_adds_to_freq_zset(self, policy, fake_redis):
        policy.register_entry("lms_cache:CS101:abc", "CS101", milvus_pk=42, ttl=86400)
        assert fake_redis.zcard(f"{REDIS_FREQ_ZSET_PREFIX}:CS101") == 1

    def test_pk_map_stored_correctly(self, policy, fake_redis):
        policy.register_entry("lms_cache:CS101:abc", "CS101", milvus_pk=99, ttl=86400)
        pk = fake_redis.get(f"{REDIS_PK_MAP_PREFIX}:lms_cache:CS101:abc")
        assert pk == "99"

    def test_pk_map_has_ttl(self, policy, fake_redis):
        policy.register_entry("lms_cache:CS101:abc", "CS101", milvus_pk=7, ttl=3600)
        ttl = fake_redis.ttl(f"{REDIS_PK_MAP_PREFIX}:lms_cache:CS101:abc")
        assert 0 < ttl <= 3600

    def test_initial_zset_score_is_near_one(self, policy, fake_redis):
        policy.register_entry("lms_cache:CS101:abc", "CS101", milvus_pk=1, ttl=86400)
        score = fake_redis.zscore(f"{REDIS_FREQ_ZSET_PREFIX}:CS101", "lms_cache:CS101:abc")
        assert 0.99 < score <= 1.0

    def test_multiple_courses_tracked_independently(self, policy, fake_redis):
        policy.register_entry("lms_cache:CS101:a", "CS101",   milvus_pk=1, ttl=86400)
        policy.register_entry("lms_cache:CS101:b", "CS101",   milvus_pk=2, ttl=86400)
        policy.register_entry("lms_cache:MATH:x",  "MATH202", milvus_pk=3, ttl=86400)

        assert fake_redis.zcard(f"{REDIS_FREQ_ZSET_PREFIX}:CS101")   == 2
        assert fake_redis.zcard(f"{REDIS_FREQ_ZSET_PREFIX}:MATH202") == 1


# ─────────────────────────────────────────────────────────────────────────────
# LFUEvictionPolicy.record_hit
# ─────────────────────────────────────────────────────────────────────────────

class TestRecordHit:

    def test_hit_count_increments(self, policy, fake_redis):
        key = "lms_cache:CS101:hit1"
        _seed_entry(fake_redis, policy, key, "CS101", milvus_pk=10, hit_count=3)

        policy.record_hit(key, "CS101")

        payload = json.loads(fake_redis.get(key))
        assert payload["hit_count"] == 4

    def test_zset_score_increases_after_hit(self, policy, fake_redis):
        key = "lms_cache:CS101:hit2"
        _seed_entry(fake_redis, policy, key, "CS101", milvus_pk=11, hit_count=1)
        score_before = fake_redis.zscore(f"{REDIS_FREQ_ZSET_PREFIX}:CS101", key)

        policy.record_hit(key, "CS101")

        score_after = fake_redis.zscore(f"{REDIS_FREQ_ZSET_PREFIX}:CS101", key)
        assert score_after > score_before

    def test_last_accessed_at_updated(self, policy, fake_redis):
        key        = "lms_cache:CS101:hit3"
        old_time   = time.time() - 3600
        _seed_entry(fake_redis, policy, key, "CS101", milvus_pk=12,
                    hit_count=1, created_at=old_time)

        policy.record_hit(key, "CS101")

        payload = json.loads(fake_redis.get(key))
        assert payload["last_accessed_at"] > old_time

    def test_missing_key_cleaned_from_zset(self, policy, fake_redis):
        key      = "lms_cache:CS101:ghost"
        freq_key = f"{REDIS_FREQ_ZSET_PREFIX}:CS101"
        fake_redis.zadd(freq_key, {key: 1.0})   # in ZSET but not in Redis

        policy.record_hit(key, "CS101")         # key doesn't exist in Redis

        assert fake_redis.zscore(freq_key, key) is None

    def test_ttl_preserved_after_hit(self, policy, fake_redis):
        key = "lms_cache:CS101:ttl"
        _seed_entry(fake_redis, policy, key, "CS101", milvus_pk=13, ttl=7200)

        policy.record_hit(key, "CS101")

        remaining = fake_redis.ttl(key)
        assert 0 < remaining <= 7200


# ─────────────────────────────────────────────────────────────────────────────
# LFUEvictionPolicy.evict_if_needed
# ─────────────────────────────────────────────────────────────────────────────

class TestEvictIfNeeded:

    def test_no_eviction_below_cap(self, policy, fake_redis):
        with patch.object(policy._redis.__class__, '__init__', return_value=None), \
             patch("src.cache.eviction.settings") as mock_settings:
            mock_settings.cache_max_entries_per_course = 100
            mock_settings.cache_eviction_batch         = 10
            mock_settings.lfu_decay_rate               = 0.01

            for i in range(5):
                _seed_entry(fake_redis, policy, f"lms_cache:CS101:k{i}", "CS101", i)

            # Patch settings on the policy itself
        with patch("src.cache.eviction.settings") as s:
            s.cache_max_entries_per_course = 100
            s.cache_eviction_batch         = 10
            s.lfu_decay_rate               = 0.01
            evicted = policy.evict_if_needed("CS101")

        assert evicted == 0

    def test_eviction_triggers_at_capacity(self, policy, fake_redis, mock_milvus):
        with patch("src.cache.eviction.settings") as s:
            s.cache_max_entries_per_course = 5
            s.cache_eviction_batch         = 2
            s.lfu_decay_rate               = 0.01

            for i in range(5):
                _seed_entry(fake_redis, policy, f"lms_cache:CS101:k{i}", "CS101", i)

            evicted = policy.evict_if_needed("CS101")

        assert evicted == 2

    def test_lowest_scored_entries_evicted_first(self, policy, fake_redis, mock_milvus):
        """
        Seed 5 entries with explicitly different scores.
        The 2 lowest should be evicted.
        """
        now = time.time()
        entries = [
            ("lms_cache:CS101:cold_old",  1,  now - 360000),   # lowest score
            ("lms_cache:CS101:cold_new",  1,  now - 100),      # second lowest
            ("lms_cache:CS101:mid",       5,  now - 3600),
            ("lms_cache:CS101:hot_old",   20, now - 36000),
            ("lms_cache:CS101:hot_new",   20, now - 100),      # highest score
        ]
        for key, hits, created in entries:
            _seed_entry(fake_redis, policy, key, "CS101",
                        milvus_pk=hash(key) % 1000,
                        hit_count=hits, created_at=created)

        with patch("src.cache.eviction.settings") as s:
            s.cache_max_entries_per_course = 5
            s.cache_eviction_batch         = 2
            s.lfu_decay_rate               = 0.01

            policy.evict_if_needed("CS101")

        freq_key      = f"{REDIS_FREQ_ZSET_PREFIX}:CS101"
        remaining     = set(fake_redis.zrange(freq_key, 0, -1))
        evicted_keys  = {"lms_cache:CS101:cold_old", "lms_cache:CS101:cold_new"}

        assert not evicted_keys & remaining, \
            f"Expected lowest-scored entries evicted, but found: {evicted_keys & remaining}"

    def test_evicted_entries_removed_from_redis(self, policy, fake_redis, mock_milvus):
        with patch("src.cache.eviction.settings") as s:
            s.cache_max_entries_per_course = 1
            s.cache_eviction_batch         = 1
            s.lfu_decay_rate               = 0.01

            key = "lms_cache:CS101:to_evict"
            _seed_entry(fake_redis, policy, key, "CS101", milvus_pk=55)
            policy.evict_if_needed("CS101")

        assert fake_redis.get(key) is None
        assert fake_redis.zscore(f"{REDIS_FREQ_ZSET_PREFIX}:CS101", key) is None

    def test_milvus_delete_called_for_evicted_entries(self, policy, fake_redis, mock_milvus):
        with patch("src.cache.eviction.settings") as s:
            s.cache_max_entries_per_course = 1
            s.cache_eviction_batch         = 1
            s.lfu_decay_rate               = 0.01

            _seed_entry(fake_redis, policy, "lms_cache:CS101:vec", "CS101", milvus_pk=777)
            policy.evict_if_needed("CS101")

        mock_milvus.delete.assert_called_once()
        call_kwargs = mock_milvus.delete.call_args[1]
        assert 777 in call_kwargs["ids"]

    def test_milvus_failure_aborts_redis_deletion(self, policy, fake_redis, mock_milvus):
        """If Milvus delete fails, Redis entry must stay intact (consistency guarantee)."""
        mock_milvus.delete.side_effect = Exception("Milvus is down")

        with patch("src.cache.eviction.settings") as s:
            s.cache_max_entries_per_course = 1
            s.cache_eviction_batch         = 1
            s.lfu_decay_rate               = 0.01

            key = "lms_cache:CS101:safe"
            _seed_entry(fake_redis, policy, key, "CS101", milvus_pk=88)
            policy.evict_if_needed("CS101")

        # Redis entry must still exist because Milvus failed
        assert fake_redis.get(key) is not None


# ─────────────────────────────────────────────────────────────────────────────
# LFUEvictionPolicy.get_course_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestCourseStats:

    def test_stats_returns_correct_count(self, policy, fake_redis):
        for i in range(3):
            _seed_entry(fake_redis, policy, f"lms_cache:CS101:s{i}", "CS101", i)

        with patch("src.cache.eviction.settings") as s:
            s.cache_max_entries_per_course = 1000
            s.lfu_decay_rate               = 0.01
            stats = policy.get_course_stats("CS101")

        assert stats["entry_count"] == 3
        assert stats["max_entries"] == 1000

    def test_utilisation_calculated_correctly(self, policy, fake_redis):
        for i in range(4):
            _seed_entry(fake_redis, policy, f"lms_cache:CS101:u{i}", "CS101", i)

        with patch("src.cache.eviction.settings") as s:
            s.cache_max_entries_per_course = 100
            s.lfu_decay_rate               = 0.01
            stats = policy.get_course_stats("CS101")

        assert stats["utilisation_pct"] == pytest.approx(4.0, abs=0.1)

    def test_empty_course_returns_zero_stats(self, policy):
        with patch("src.cache.eviction.settings") as s:
            s.cache_max_entries_per_course = 1000
            s.lfu_decay_rate               = 0.01
            stats = policy.get_course_stats("EMPTY_COURSE")

        assert stats["entry_count"]    == 0
        assert stats["utilisation_pct"] == 0.0
        assert stats["top_entries"]    == []
        assert stats["bottom_entries"] == []


# ─────────────────────────────────────────────────────────────────────────────
# LFUEvictionPolicy.clear_course
# ─────────────────────────────────────────────────────────────────────────────

class TestClearCourse:

    def test_clears_all_entries(self, policy, fake_redis, mock_milvus):
        for i in range(4):
            _seed_entry(fake_redis, policy, f"lms_cache:CS101:c{i}", "CS101", i)

        count = policy.clear_course("CS101")

        assert count == 4
        assert fake_redis.zcard(f"{REDIS_FREQ_ZSET_PREFIX}:CS101") == 0

    def test_clear_does_not_affect_other_courses(self, policy, fake_redis, mock_milvus):
        _seed_entry(fake_redis, policy, "lms_cache:CS101:x",   "CS101",   1)
        _seed_entry(fake_redis, policy, "lms_cache:MATH:y",    "MATH202", 2)

        policy.clear_course("CS101")

        assert fake_redis.zcard(f"{REDIS_FREQ_ZSET_PREFIX}:MATH202") == 1


# ─────────────────────────────────────────────────────────────────────────────
# Integration: CacheManager with LFU
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheManagerWithLFU:
    """
    Integration tests for CacheManager.update_cache() and search_cache()
    with real LFU tracking (fakeredis + mocked Milvus + mocked embedder).
    """

    def _make_manager(self, fake_redis):
        from src.cache.cache_manager import CacheManager

        mock_milvus  = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 1024

        # Simulate Milvus insert returning a primary key
        insert_result = SimpleNamespace(primary_keys=[12345])
        mock_milvus.insert.return_value   = insert_result
        mock_milvus.flush.return_value    = None
        mock_milvus.delete.return_value   = None

        # Simulate search returning a hit
        hit_entity = {"entity": {"redis_key": "some_key", "course_tag": "CS101"}}
        mock_milvus.search.return_value = [[SimpleNamespace(**{"get": lambda k, d=None: {"distance": 0.99}.get(k, d), "entity": hit_entity["entity"]})]]

        manager = CacheManager.__new__(CacheManager)
        manager._milvus   = mock_milvus
        manager._redis    = fake_redis
        manager._embedder = mock_embedder
        manager._eviction = LFUEvictionPolicy(redis=fake_redis, milvus=mock_milvus)

        return manager, mock_milvus

    def test_update_cache_registers_entry_in_zset(self, fake_redis):
        manager, _ = self._make_manager(fake_redis)

        with patch("src.cache.eviction.settings") as s:
            s.cache_max_entries_per_course = 1000
            s.cache_eviction_batch         = 10
            s.lfu_decay_rate               = 0.01
            with patch("src.cache.cache_manager.settings") as cs:
                cs.similarity_threshold = 0.92
                cs.cache_ttl_seconds    = 86400
                cs.cache_max_entries_per_course = 1000
                cs.cache_eviction_batch         = 10
                cs.lfu_decay_rate               = 0.01
                manager.update_cache("What is ML?", "ML is...", "CS101")

        freq_key = f"{REDIS_FREQ_ZSET_PREFIX}:CS101"
        assert fake_redis.zcard(freq_key) == 1

    def test_update_cache_payload_has_hit_count_zero(self, fake_redis):
        manager, _ = self._make_manager(fake_redis)

        with patch("src.cache.eviction.settings") as s, \
             patch("src.cache.cache_manager.settings") as cs:
            s.cache_max_entries_per_course  = 1000
            s.cache_eviction_batch          = 10
            s.lfu_decay_rate                = 0.01
            cs.cache_ttl_seconds            = 86400
            cs.cache_max_entries_per_course = 1000
            cs.cache_eviction_batch         = 10
            cs.lfu_decay_rate               = 0.01
            cs.similarity_threshold         = 0.92

            key = manager.update_cache("What is ML?", "ML is...", "CS101")

        payload = json.loads(fake_redis.get(key))
        assert payload["hit_count"] == 0
        assert "created_at" in payload
        assert "last_accessed_at" in payload