import logging

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_cache_manager, get_kb_manager
from src.api.models import QueryRequest, QueryResponse
from src.cache.cache_manager import CacheManager
from src.knowledge.kb_manager import KBManager, CourseNotFoundError
from src.orchestrator import process_lms_query

router = APIRouter(prefix="/api/v1", tags=["Query"])
logger = logging.getLogger(__name__)


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a course query",
    description=(
        "Submit a student question for a given course. "
        "Returns a cached answer if a semantically similar query was seen before "
        "(similarity ≥ threshold), otherwise retrieves context from the knowledge "
        "base, calls the LLM, caches the result, and returns the fresh answer."
    ),
    responses={
        200: {"description": "Answer returned (cache hit or fresh LLM response)"},
        404: {"description": "course_tag not initialised — call POST /api/v1/courses first"},
        422: {"description": "Request validation error"},
        500: {"description": "Internal server error"},
    },
)
async def query(
    body: QueryRequest,
    cache_manager: CacheManager = Depends(get_cache_manager),
    kb_manager: KBManager = Depends(get_kb_manager),
) -> QueryResponse:

    logger.info(
        "Query received | course=%s | query='%.80s'",
        body.course_tag, body.query,
    )

    # Pre-flight: verify the course partition exists before spinning up embeddings
    if not kb_manager.partition_exists(body.course_tag):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Course '{body.course_tag}' is not initialised. "
                "Call POST /api/v1/courses to register it first."
            ),
        )

    result = process_lms_query(
        user_query=body.query,
        course_tag=body.course_tag,
        similarity_threshold=body.similarity_threshold,
        top_k_kb=body.top_k,
    )

    if not result.ok:
        # CourseNotFoundError surfaces here as a 404
        if "not initialised" in (result.error or "").lower() or \
           "no partition" in (result.error or "").lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.error,
            )
        # Everything else is a 500
        logger.error("Orchestrator error | course=%s | error=%s", body.course_tag, result.error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error,
        )

    return QueryResponse(
        answer=result.answer,
        course_tag=result.course_tag,
        cache_hit=result.cache_hit,
        similarity=result.similarity,
        kb_chunks_used=result.kb_chunks_used,
        latency_ms=result.latency_ms,
    )
