import re
import uuid
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_request_id() -> str:
    return uuid.uuid4().hex


_COURSE_TAG_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _validate_course_tag(v: str) -> str:
    v = v.strip().upper()
    if not _COURSE_TAG_RE.match(v):
        raise ValueError(
            f"course_tag '{v}' is invalid. "
            "Use only letters, digits, underscores, and hyphens (max 64 chars)."
        )
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Query endpoint  POST /api/v1/query
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The student's question.",
        examples=["What is gradient descent?"],
    )
    course_tag: str = Field(
        ...,
        description="Course identifier. Must match an initialised Milvus partition.",
        examples=["CS101"],
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override the global similarity threshold for this request only.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of KB chunks to retrieve on a cache miss.",
    )

    @field_validator("course_tag")
    @classmethod
    def validate_course_tag(cls, v: str) -> str:
        return _validate_course_tag(v)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class QueryResponse(BaseModel):
    request_id: str = Field(default_factory=_make_request_id)
    answer: str
    course_tag: str
    cache_hit: bool
    similarity: Optional[float] = Field(
        default=None,
        description="Cosine similarity score. Present only on cache hit.",
    )
    kb_chunks_used: int = Field(
        default=0,
        description="Number of KB chunks assembled as LLM context. 0 on cache hit.",
    )
    latency_ms: float
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Course management  GET/POST /api/v1/courses
# ─────────────────────────────────────────────────────────────────────────────

class CreateCourseRequest(BaseModel):
    course_tag: str = Field(
        ...,
        description="Unique course identifier. Creates a Milvus partition.",
        examples=["PHYS404"],
    )

    @field_validator("course_tag")
    @classmethod
    def validate_course_tag(cls, v: str) -> str:
        return _validate_course_tag(v)


class CourseItem(BaseModel):
    course_tag: str
    partition_exists: bool


class CourseListResponse(BaseModel):
    request_id: str = Field(default_factory=_make_request_id)
    courses: list[CourseItem]
    total: int


class CreateCourseResponse(BaseModel):
    request_id: str = Field(default_factory=_make_request_id)
    course_tag: str
    created: bool
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Cache management  DELETE /api/v1/cache/{course_tag}
# ─────────────────────────────────────────────────────────────────────────────

class CacheInvalidateResponse(BaseModel):
    request_id: str = Field(default_factory=_make_request_id)
    course_tag: str
    keys_deleted: int
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Health  GET /health
# ─────────────────────────────────────────────────────────────────────────────

class ServiceStatus(BaseModel):
    status: str           # "ok" | "degraded" | "down"
    latency_ms: float


class HealthResponse(BaseModel):
    status: str           # "healthy" | "degraded" | "unhealthy"
    version: str
    milvus: ServiceStatus
    redis: ServiceStatus


# ─────────────────────────────────────────────────────────────────────────────
# Generic error  (returned bsy the exception handler)
# ─────────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    request_id: str = Field(default_factory=_make_request_id)
    error: str
    detail: Optional[str] = None