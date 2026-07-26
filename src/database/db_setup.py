import logging
import sys
from typing import Optional

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusException

from config.constants import (
    CACHE_COLLECTION_NAME,
    CACHE_VECTOR_INDEX,
    CHUNK_TEXT_MAX_LEN,
    COURSE_TAG_MAX_LEN,
    DEFAULT_KB_PARTITION,
    FIELD_CHUNK_TEXT,
    FIELD_CHUNK_VECTOR,
    FIELD_COURSE_TAG,
    FIELD_METADATA,
    FIELD_PK,
    FIELD_QUERY_VECTOR,
    FIELD_REDIS_KEY,
    KB_COLLECTION_NAME,
    KB_VECTOR_INDEX,
    METADATA_MAX_LEN,
    REDIS_KEY_MAX_LEN,
)
from config.settings import settings
from src.database.milvus_client import get_milvus_client
from src.database.redis_client import get_redis_client

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Schema builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_cache_schema() -> CollectionSchema:
    """
    lms_semantic_cache schema:
      pk            INT64 auto-id   — primary key
      query_vector  FLOAT_VECTOR    — embedded user query
      course_tag    VARCHAR indexed — scalar pre-filter for course isolation
      redis_key     VARCHAR         — pointer to Redis payload
    """
    fields = [
        FieldSchema(
            name=FIELD_PK,
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        ),
        FieldSchema(
            name=FIELD_QUERY_VECTOR,
            dtype=DataType.FLOAT_VECTOR,
            dim=settings.embedding_dim,
        ),
        FieldSchema(
            name=FIELD_COURSE_TAG,
            dtype=DataType.VARCHAR,
            max_length=COURSE_TAG_MAX_LEN,
        ),
        FieldSchema(
            name=FIELD_REDIS_KEY,
            dtype=DataType.VARCHAR,
            max_length=REDIS_KEY_MAX_LEN,
        ),
    ]
    return CollectionSchema(
        fields=fields,
        description="Semantic cache: maps query embeddings to Redis keys, scoped by course.",
        enable_dynamic_field=False,
    )


def _build_kb_schema() -> CollectionSchema:
    """
    course_knowledge_base schema:
      pk            INT64 auto-id   — primary key
      chunk_vector  FLOAT_VECTOR    — embedded knowledge chunk
      course_tag    VARCHAR indexed — mirrors partition name; enables cross-partition queries
      chunk_text    VARCHAR         — raw text returned for RAG context
      metadata      VARCHAR         — JSON: {source, page_number, section, ingested_at}
    """
    fields = [
        FieldSchema(
            name=FIELD_PK,
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        ),
        FieldSchema(
            name=FIELD_CHUNK_VECTOR,
            dtype=DataType.FLOAT_VECTOR,
            dim=settings.embedding_dim,
        ),
        FieldSchema(
            name=FIELD_COURSE_TAG,
            dtype=DataType.VARCHAR,
            max_length=COURSE_TAG_MAX_LEN,
        ),
        FieldSchema(
            name=FIELD_CHUNK_TEXT,
            dtype=DataType.VARCHAR,
            max_length=CHUNK_TEXT_MAX_LEN,
        ),
        FieldSchema(
            name=FIELD_METADATA,
            dtype=DataType.VARCHAR,
            max_length=METADATA_MAX_LEN,
        ),
    ]
    return CollectionSchema(
        fields=fields,
        description="Knowledge base: course content partitioned by course_tag.",
        enable_dynamic_field=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Collection setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def _setup_cache_collection(client) -> None:
    """Create lms_semantic_cache and its indexes if they don't exist."""

    if not client.has_collection(CACHE_COLLECTION_NAME):
        logger.info("Creating collection '%s' …", CACHE_COLLECTION_NAME)
        client.create_collection(
            collection_name=CACHE_COLLECTION_NAME,
            schema=_build_cache_schema(),
            consistency_level="Bounded",
        )
        logger.info("Collection '%s' created.", CACHE_COLLECTION_NAME)
    else:
        logger.info("Collection '%s' already exists.", CACHE_COLLECTION_NAME)

    cache_index_params = client.prepare_index_params()

    cache_index_params.add_index(
        field_name=FIELD_QUERY_VECTOR,
        index_name="idx_cache_vector_hnsw",
        index_type=CACHE_VECTOR_INDEX["index_type"],
        metric_type=CACHE_VECTOR_INDEX["metric_type"],
        params=CACHE_VECTOR_INDEX["params"],
    )

    cache_index_params.add_index(
        field_name=FIELD_COURSE_TAG,
        index_name="idx_cache_course_tag",
        index_type="INVERTED",
    )

    client.create_index(
        collection_name=CACHE_COLLECTION_NAME,
        index_params=cache_index_params,
    )
    logger.info("Cache collection indexes ready.")


def _setup_kb_collection(client, initial_course_tags: Optional[list[str]] = None) -> None:
    """
    Create course_knowledge_base and its indexes.
    Optionally pre-create partitions for known courses.
    """

    if not client.has_collection(KB_COLLECTION_NAME):
        logger.info("Creating collection '%s' …", KB_COLLECTION_NAME)
        client.create_collection(
            collection_name=KB_COLLECTION_NAME,
            schema=_build_kb_schema(),
            # Strong consistency for KB — ingestion must be immediately visible
            consistency_level="Strong",
        )
        logger.info("Collection '%s' created.", KB_COLLECTION_NAME)
    else:
        logger.info("Collection '%s' already exists.", KB_COLLECTION_NAME)

    # ── Build IndexParams object (required by pymilvus >= 2.4 MilvusClient) ──
    kb_index_params = client.prepare_index_params()

    # IVF_FLAT vector index
    # Chosen for the KB because:
    #   • nprobe dial gives recall/speed trade-off tunable at query time.
    #   • Scales better than HNSW for large collections (100k–1M+ chunks).
    #   • Partition-level search narrows candidates before IVF centroids are probed.
    kb_index_params.add_index(
        field_name=FIELD_CHUNK_VECTOR,
        index_name="idx_kb_chunk_vector_ivf",
        index_type=KB_VECTOR_INDEX["index_type"],
        metric_type=KB_VECTOR_INDEX["metric_type"],
        params=KB_VECTOR_INDEX["params"],
    )

    # INVERTED scalar index on course_tag (same rationale as cache collection)
    kb_index_params.add_index(
        field_name=FIELD_COURSE_TAG,
        index_name="idx_kb_course_tag",
        index_type="INVERTED",
    )

    client.create_index(
        collection_name=KB_COLLECTION_NAME,
        index_params=kb_index_params,
    )

    # ── Partition management ──────────────────────────────────────────────────
    # list_partitions() returns list[str] in pymilvus 2.4+ MilvusClient
    existing = set(client.list_partitions(KB_COLLECTION_NAME))

    courses_to_create = set(initial_course_tags or []) | {DEFAULT_KB_PARTITION}
    for tag in courses_to_create:
        if tag not in existing:
            client.create_partition(collection_name=KB_COLLECTION_NAME, partition_name=tag)
            logger.info("Created KB partition: '%s'", tag)
        else:
            logger.debug("KB partition '%s' already exists.", tag)

    logger.info("Knowledge base collection indexes and partitions ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def create_kb_partition(course_tag: str) -> None:
    """
    Dynamically add a partition for a new course.
    Call this whenever a new course is onboarded to the LMS.

    Args:
        course_tag: Unique course identifier, e.g. "PHYS404_2025".
    """
    client = get_milvus_client()
    existing = set(client.list_partitions(KB_COLLECTION_NAME))
    if course_tag in existing:
        logger.info("Partition '%s' already exists — skipping.", course_tag)
        return
    client.create_partition(collection_name=KB_COLLECTION_NAME, partition_name=course_tag)
    logger.info("Dynamically created KB partition for course: '%s'", course_tag)


def verify_redis_connection() -> None:
    """Raise RuntimeError if Redis is unreachable."""
    redis_client = get_redis_client()
    if not redis_client.ping():
        raise RuntimeError(
            f"Redis is unreachable at '{settings.redis_url}'. "
            "Verify the server is running and REDIS_URL in .env is correct."
        )
    logger.info("Redis connection verified ✓")


def teardown_all_data() -> None:
    """
    Nuclear reset — drops BOTH Milvus collections and flushes all
    lms_cache:* keys from Redis.

    USE CASES
    ---------
    - Before running the test suite (guarantees a cold-start state)
    - Manual reset during development

    DO NOT call in production without a deliberate data-loss decision.

    After calling this, you MUST call setup_all_databases() to recreate
    the collections and indexes before the application can run again.
    """
    client      = get_milvus_client()
    redis_client = get_redis_client()

    logger.warning("=== TEARDOWN: dropping all collections and cache data ===")

    # ── Drop Milvus collections ───────────────────────────────────────────────
    for collection_name in (CACHE_COLLECTION_NAME, KB_COLLECTION_NAME):
        if client.has_collection(collection_name):
            # release_collection first — prevents "collection loaded" errors on drop
            try:
                client.release_collection(collection_name)
            except MilvusException:
                pass   # already released or never loaded — safe to ignore
            client.drop_collection(collection_name)
            logger.warning("Dropped Milvus collection: '%s'", collection_name)
        else:
            logger.info("Collection '%s' did not exist — skipping drop.", collection_name)

    # ── Flush Redis cache keys (lms_cache:*) ─────────────────────────────────
    # We use SCAN + DELETE in batches rather than FLUSHDB so we only touch
    # our own namespace and don't nuke unrelated Redis data.
    from config.constants import REDIS_CACHE_PREFIX
    pattern   = f"{REDIS_CACHE_PREFIX}:*"
    cursor    = 0
    total_del = 0
    while True:
        cursor, keys = redis_client.scan(cursor=cursor, match=pattern, count=200)
        if keys:
            redis_client.delete(*keys)
            total_del += len(keys)
        if cursor == 0:
            break
    logger.warning("Deleted %d Redis cache keys matching '%s'", total_del, pattern)

    logger.warning("=== TEARDOWN COMPLETE — call setup_all_databases() to rebuild ===")


def setup_all_databases(initial_course_tags: Optional[list[str]] = None) -> None:
    """
    Master setup — call once on application startup.

    Args:
        initial_course_tags: Course IDs to pre-partition in the KB collection.
                             If None, only the default partition is created.

    Raises:
        RuntimeError:    If Redis is unreachable.
        MilvusException: If Milvus is unreachable or schema conflicts.
    """
    logger.info("=== LMS Semantic Cache — Database Setup START ===")

    verify_redis_connection()           # fail fast if Redis is down
    client = get_milvus_client()        # triggers Milvus connection

    _setup_cache_collection(client)
    _setup_kb_collection(client, initial_course_tags=initial_course_tags)

    # Load both collections into Milvus memory for querying (idempotent)
    client.load_collection(CACHE_COLLECTION_NAME)
    client.load_collection(KB_COLLECTION_NAME)

    logger.info("=== LMS Semantic Cache — Database Setup COMPLETE ===")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint: python -m src.database.db_setup CS101 MATH202
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    course_tags = sys.argv[1:] or None
    setup_all_databases(initial_course_tags=course_tags)