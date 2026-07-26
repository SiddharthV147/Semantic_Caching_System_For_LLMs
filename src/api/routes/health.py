import time
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config.settings import settings
from src.api.models import HealthResponse, ServiceStatus
from src.database.milvus_client import get_milvus_client
from src.database.redis_client import get_redis_client

router = APIRouter(tags=["Health"])
logger = logging.getLogger(__name__)


def _check_milvus() -> ServiceStatus:
    t = time.perf_counter()
    try:
        client = get_milvus_client()
        # list_collections() is the lightest read operation on Milvus
        client.list_collections()
        latency = round((time.perf_counter() - t) * 1000, 2)
        return ServiceStatus(status="ok", latency_ms=latency)
    except Exception as exc:
        logger.warning("Milvus health check failed: %s", exc)
        latency = round((time.perf_counter() - t) * 1000, 2)
        return ServiceStatus(status="down", latency_ms=latency)


def _check_redis() -> ServiceStatus:
    t = time.perf_counter()
    try:
        get_redis_client().ping()
        latency = round((time.perf_counter() - t) * 1000, 2)
        return ServiceStatus(status="ok", latency_ms=latency)
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        latency = round((time.perf_counter() - t) * 1000, 2)
        return ServiceStatus(status="down", latency_ms=latency)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Infrastructure health check",
    description=(
        "Probes Milvus and Redis. Returns HTTP 200 if all services are "
        "reachable, HTTP 503 if any service is down."
    ),
)
async def health_check() -> JSONResponse:
    milvus_status = _check_milvus()
    redis_status  = _check_redis()

    all_ok = milvus_status.status == "ok" and redis_status.status == "ok"
    overall = "healthy" if all_ok else "unhealthy"
    http_status = 200 if all_ok else 503

    body = HealthResponse(
        status=overall,
        version=settings.api_version,
        milvus=milvus_status,
        redis=redis_status,
    )
    return JSONResponse(content=body.model_dump(), status_code=http_status)
