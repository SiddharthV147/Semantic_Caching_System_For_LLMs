import logging

from fastapi import APIRouter, Depends, HTTPException, status

from config.constants import KB_COLLECTION_NAME
from src.api.dependencies import get_kb_manager
from src.api.models import (
    CreateCourseRequest,
    CreateCourseResponse,
    CourseItem,
    CourseListResponse,
)
from src.database.db_setup import create_kb_partition
from src.database.milvus_client import get_milvus_client
from src.knowledge.kb_manager import KBManager

router = APIRouter(prefix="/api/v1/courses", tags=["Courses"])
logger = logging.getLogger(__name__)

# Milvus always creates this partition internally — exclude from user-facing list
_INTERNAL_PARTITIONS = {"_default"}


@router.get(
    "",
    response_model=CourseListResponse,
    summary="List all registered courses",
    description="Returns all course partitions currently registered in the knowledge base.",
)
async def list_courses(
    kb_manager: KBManager = Depends(get_kb_manager),
) -> CourseListResponse:

    try:
        client = get_milvus_client()
        all_partitions: list[str] = client.list_partitions(KB_COLLECTION_NAME)
    except Exception as exc:
        logger.exception("Failed to list partitions from Milvus.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not fetch course list: {exc}",
        )

    courses = [
        CourseItem(course_tag=p, partition_exists=True)
        for p in sorted(all_partitions)
        if p not in _INTERNAL_PARTITIONS
    ]

    return CourseListResponse(courses=courses, total=len(courses))


@router.post(
    "",
    response_model=CreateCourseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new course",
    description=(
        "Creates a dedicated Milvus partition for the course. "
        "Idempotent — returns HTTP 200 (not 201) if the course already exists."
    ),
)
async def create_course(
    body: CreateCourseRequest,
    kb_manager: KBManager = Depends(get_kb_manager),
) -> CreateCourseResponse:

    already_exists = kb_manager.partition_exists(body.course_tag)

    if already_exists:
        logger.info("Course '%s' already exists — skipping creation.", body.course_tag)
        return CreateCourseResponse(
            course_tag=body.course_tag,
            created=False,
            message=f"Course '{body.course_tag}' already registered.",
        )

    try:
        create_kb_partition(body.course_tag)
        logger.info("Course '%s' registered successfully.", body.course_tag)
    except Exception as exc:
        logger.exception("Failed to create partition for course '%s'.", body.course_tag)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not create course partition: {exc}",
        )

    return CreateCourseResponse(
        course_tag=body.course_tag,
        created=True,
        message=f"Course '{body.course_tag}' registered successfully.",
    )
