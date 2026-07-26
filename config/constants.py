"""
config/constants.py
Hard-coded names and index parameters.
"""

# ── Collection / partition names ─────────────────────────────────────────────
CACHE_COLLECTION_NAME   = "lms_semantic_cache"
KB_COLLECTION_NAME      = "course_knowledge_base"
DEFAULT_KB_PARTITION    = "_default"

# ── Cache collection field names ─────────────────────────────────────────────
FIELD_PK            = "pk"
FIELD_QUERY_VECTOR  = "query_vector"
FIELD_COURSE_TAG    = "course_tag"
FIELD_REDIS_KEY     = "redis_key"

# ── KB collection field names ─────────────────────────────────────────────────
FIELD_CHUNK_VECTOR  = "chunk_vector"
FIELD_CHUNK_TEXT    = "chunk_text"
FIELD_METADATA      = "metadata"

# ── Redis key namespaces ──────────────────────────────────────────────────────
REDIS_CACHE_PREFIX       = "lms_cache"        # lms_cache:{course_tag}:{uuid}  → JSON payload
REDIS_FREQ_ZSET_PREFIX   = "lms_freq"         # lms_freq:{course_tag}           → ZSET of scores
REDIS_PK_MAP_PREFIX      = "lms_pk"           # lms_pk:{redis_key}              → Milvus pk (int)

# ── Schema field length limits ────────────────────────────────────────────────
COURSE_TAG_MAX_LEN  = 64
REDIS_KEY_MAX_LEN   = 256
CHUNK_TEXT_MAX_LEN  = 4096
METADATA_MAX_LEN    = 1024

# ── Eviction ──────────────────────────────────────────────────────────────────
# Maximum cache entries stored per course before eviction kicks in.
# At 1000 entries per course and 1024-dim float32 vectors,
# Milvus RAM cost is ~4MB per course — well within limits.
DEFAULT_MAX_ENTRIES_PER_COURSE = 1000

# LFU decay: score = hit_count / (1 + age_hours * DECAY_RATE)
# DECAY_RATE=0.01 means an entry's effective score halves every ~70 hours.
# This prevents high-frequency old entries from permanently blocking new ones.
LFU_DECAY_RATE = 0.01

# ── Vector index — HNSW (speed-optimised for hot cache path) ─────────────────
CACHE_VECTOR_INDEX = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200},
}

# ── Vector index — IVF_FLAT (recall-optimised for large KB) ──────────────────
KB_VECTOR_INDEX = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128},
}

# ── Search parameters ─────────────────────────────────────────────────────────
CACHE_SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"ef": 64}}
KB_SEARCH_PARAMS    = {"metric_type": "COSINE", "params": {"nprobe": 16}}

# ── Default top-k values ──────────────────────────────────────────────────────
TOP_K_CACHE = 1
TOP_K_KB    = 5