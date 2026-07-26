import logging
import re

from fastapi import APIRouter, HTTPException, Path, status

from config.constants import REDIS_CACHE_PREFIX
from src.api.models import CacheInvalidateResponse
from src.database.redis_client import get_redis_client

router = APIRouter(prefix="/api/v1/cache", tags=["Cache"])
logger = logging.getLogger(__name__)

_COURSE_TAG_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


@router.delete(
    "/{course_tag}",
    response_model=CacheInvalidateResponse,
    summary="Invalidate cache for a course",
    description=(
        "Deletes all Redis cache entries for the given course. "
        "The next queries for this course will be cache misses and go through "
        "KB retrieval + LLM. Milvus vectors are not deleted — they are orphaned "
        "and will be skipped (Redis key missing = treated as stale miss)."
    ),
    responses={
        200: {"description": "Cache entries deleted (may be 0 if cache was empty)"},
        400: {"description": "Invalid course_tag format"},
        500: {"description": "Redis error"},
    },
)
async def invalidate_course_cache(
    course_tag: str = Path(
        ...,
        description="Course identifier whose cache entries will be deleted.",
        examples=["CS101"],
    ),
) -> CacheInvalidateResponse:

    course_tag = course_tag.strip().upper()

    if not _COURSE_TAG_RE.match(course_tag):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid course_tag '{course_tag}'. Use letters, digits, underscores, hyphens only.",
        )

    pattern = f"{REDIS_CACHE_PREFIX}:{course_tag}:*"

    try:
        redis = get_redis_client()
        cursor    = 0
        total_del = 0

        while True:
            cursor, keys = redis.scan(cursor=cursor, match=pattern, count=200)
            if keys:
                redis.delete(*keys)
                total_del += len(keys)
            if cursor == 0:
                break

    except Exception as exc:
        logger.exception("Redis error during cache invalidation for course '%s'.", course_tag)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Redis error: {exc}",
        )

    logger.info("Invalidated %d cache entries for course '%s'.", total_del, course_tag)

    return CacheInvalidateResponse(
        course_tag=course_tag,
        keys_deleted=total_del,
        message=(
            f"Deleted {total_del} cache entries for '{course_tag}'."
            if total_del > 0
            else f"No cache entries found for '{course_tag}' — nothing deleted."
        ),
    )
