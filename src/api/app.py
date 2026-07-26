import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import settings
from src.api.models import ErrorResponse
from src.api.routes import health, query, courses, cache
from src.cache.cache_manager import CacheManager
from src.database.db_setup import setup_all_databases
from src.database.milvus_client import _manager as milvus_manager
from src.database.redis_client import _manager as redis_manager
from src.embeddings.embedding_service import get_embedder
from src.knowledge.kb_manager import KBManager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Runs once at startup (before first request) and once at shutdown.
    FastAPI guarantees this even if startup raises — cleanup always runs.
    """
    # ── STARTUP ───────────────────────────────────────────────────────────────
    logger.info("=== API startup: initialising services ===")

    # 1. Ensure Milvus collections exist and are loaded
    setup_all_databases()

    # 2. Warm the embedding model — avoids a cold-start penalty on the first request
    embedder = get_embedder()
    embedder.embed("warmup")
    logger.info("Embedding model warmed up.")

    # 3. Attach managers to app.state so routes can access them via Depends()
    app.state.cache_manager = CacheManager()
    app.state.kb_manager    = KBManager()

    logger.info("=== API startup complete — ready to serve ===")

    yield  # server is running here

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("=== API shutdown: releasing connections ===")
    milvus_manager.close()
    redis_manager.close()
    logger.info("=== API shutdown complete ===")


def create_app() -> FastAPI:
    """
    Application factory. Returns a configured FastAPI instance.
    Called by main.py and by the test suite (with overridden deps).
    """
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=(
            "Course-aware semantic cache for LMS. "
            "Reduces LLM calls by serving semantically similar cached answers, "
            "with strict per-course data isolation."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Tighten origins in production via environment variable
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(courses.router)
    app.include_router(cache.router)

    # ── Global exception handlers ─────────────────────────────────────────────

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url)
        body = ErrorResponse(error="Internal server error", detail=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=body.model_dump(),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(
        request: Request, exc: ValueError
    ) -> JSONResponse:
        body = ErrorResponse(error="Validation error", detail=str(exc))
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=body.model_dump(),
        )

    return app


# Module-level app instance — used by uvicorn in main.py
app = create_app()