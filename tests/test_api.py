import pytest
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_cache_manager, get_kb_manager
from src.cache.cache_manager import CacheHit, CacheMiss
from src.orchestrator import LMSQueryResponse


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_cache_manager() -> MagicMock:
    """Stateful mock CacheManager — configure per-test via mock.method.return_value."""
    return MagicMock()


@pytest.fixture(scope="module")
def mock_kb_manager() -> MagicMock:
    """Stateful mock KBManager."""
    m = MagicMock()
    # Default: every course exists
    m.partition_exists.return_value = True
    return m


@pytest.fixture(scope="module")
def client(mock_cache_manager, mock_kb_manager) -> TestClient:
    """
    Build the FastAPI app with:
    - lifespan replaced by a no-op (no DB calls on startup)
    - managers replaced by mocks via dependency_overrides
    """

    @asynccontextmanager
    async def noop_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Inject mocks into app.state exactly as the real lifespan does
        app.state.cache_manager = mock_cache_manager
        app.state.kb_manager    = mock_kb_manager
        yield

    with patch("src.api.app.lifespan", noop_lifespan):
        app = create_app()

    # Override Depends() functions with lambdas returning our mocks
    app.dependency_overrides[get_cache_manager] = lambda: mock_cache_manager
    app.dependency_overrides[get_kb_manager]    = lambda: mock_kb_manager

    # TestClient manages startup/shutdown of the app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_returns_200_when_services_up(self, client):
        with patch("src.api.routes.health._check_milvus") as m_milvus, \
             patch("src.api.routes.health._check_redis") as m_redis:

            from src.api.models import ServiceStatus
            m_milvus.return_value = ServiceStatus(status="ok", latency_ms=1.0)
            m_redis.return_value  = ServiceStatus(status="ok", latency_ms=0.5)

            resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["milvus"]["status"] == "ok"
        assert body["redis"]["status"] == "ok"
        assert "version" in body

    def test_health_returns_503_when_milvus_down(self, client):
        with patch("src.api.routes.health._check_milvus") as m_milvus, \
             patch("src.api.routes.health._check_redis") as m_redis:

            from src.api.models import ServiceStatus
            m_milvus.return_value = ServiceStatus(status="down", latency_ms=5000.0)
            m_redis.return_value  = ServiceStatus(status="ok",   latency_ms=0.5)

            resp = client.get("/health")

        assert resp.status_code == 503
        assert resp.json()["status"] == "unhealthy"

    def test_health_returns_503_when_redis_down(self, client):
        with patch("src.api.routes.health._check_milvus") as m_milvus, \
             patch("src.api.routes.health._check_redis") as m_redis:

            from src.api.models import ServiceStatus
            m_milvus.return_value = ServiceStatus(status="ok",   latency_ms=1.0)
            m_redis.return_value  = ServiceStatus(status="down", latency_ms=5000.0)

            resp = client.get("/health")

        assert resp.status_code == 503


# ─────────────────────────────────────────────────────────────────────────────
# Query endpoint — request validation
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryValidation:

    def test_missing_query_field_returns_422(self, client):
        resp = client.post("/api/v1/query", json={"course_tag": "CS101"})
        assert resp.status_code == 422

    def test_missing_course_tag_returns_422(self, client):
        resp = client.post("/api/v1/query", json={"query": "What is ML?"})
        assert resp.status_code == 422

    def test_query_too_short_returns_422(self, client):
        resp = client.post("/api/v1/query", json={"query": "hi", "course_tag": "CS101"})
        assert resp.status_code == 422

    def test_query_too_long_returns_422(self, client):
        resp = client.post("/api/v1/query", json={"query": "x" * 2001, "course_tag": "CS101"})
        assert resp.status_code == 422

    def test_invalid_course_tag_characters_returns_422(self, client):
        resp = client.post(
            "/api/v1/query",
            json={"query": "What is ML?", "course_tag": "CS 101; DROP TABLE--"},
        )
        assert resp.status_code == 422

    def test_course_tag_is_uppercased(self, client, mock_kb_manager):
        """course_tag should be normalised to uppercase by the validator."""
        mock_kb_manager.partition_exists.return_value = False
        resp = client.post(
            "/api/v1/query",
            json={"query": "What is gradient descent?", "course_tag": "cs101"},
        )
        # 404 because partition doesn't exist, but the tag was normalised
        assert resp.status_code == 404
        assert "CS101" in resp.json()["detail"]

    def test_invalid_similarity_threshold_returns_422(self, client):
        resp = client.post(
            "/api/v1/query",
            json={"query": "What is ML?", "course_tag": "CS101", "similarity_threshold": 1.5},
        )
        assert resp.status_code == 422

    def test_top_k_out_of_range_returns_422(self, client):
        resp = client.post(
            "/api/v1/query",
            json={"query": "What is ML?", "course_tag": "CS101", "top_k": 0},
        )
        assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Query endpoint — business logic
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryLogic:

    def test_unknown_course_returns_404(self, client, mock_kb_manager):
        mock_kb_manager.partition_exists.return_value = False
        resp = client.post(
            "/api/v1/query",
            json={"query": "What is gradient descent?", "course_tag": "UNKNOWN"},
        )
        assert resp.status_code == 404
        assert "not initialised" in resp.json()["detail"].lower()

    def test_cache_miss_returns_200_with_correct_shape(self, client, mock_kb_manager):
        mock_kb_manager.partition_exists.return_value = True

        mock_response = LMSQueryResponse(
            answer="Gradient descent minimises a function by iterating in the steepest descent direction.",
            course_tag="CS101",
            query_text="What is gradient descent?",
            cache_hit=False,
            similarity=None,
            latency_ms=120.5,
            kb_chunks_used=3,
        )

        with patch("src.api.routes.query.process_lms_query", return_value=mock_response):
            resp = client.post(
                "/api/v1/query",
                json={"query": "What is gradient descent?", "course_tag": "CS101"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["cache_hit"] is False
        assert body["kb_chunks_used"] == 3
        assert body["similarity"] is None
        assert "gradient descent" in body["answer"].lower()
        assert "request_id" in body
        assert "latency_ms" in body

    def test_cache_hit_returns_200_with_similarity(self, client, mock_kb_manager):
        mock_kb_manager.partition_exists.return_value = True

        mock_response = LMSQueryResponse(
            answer="Gradient descent minimises a function by iterating in the steepest descent direction.",
            course_tag="CS101",
            query_text="What is gradient descent?",
            cache_hit=True,
            similarity=0.9873,
            latency_ms=18.3,
            kb_chunks_used=0,
        )

        with patch("src.api.routes.query.process_lms_query", return_value=mock_response):
            resp = client.post(
                "/api/v1/query",
                json={"query": "What is gradient descent?", "course_tag": "CS101"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["cache_hit"] is True
        assert body["similarity"] == pytest.approx(0.9873, abs=1e-4)
        assert body["kb_chunks_used"] == 0

    def test_orchestrator_error_returns_500(self, client, mock_kb_manager):
        mock_kb_manager.partition_exists.return_value = True

        error_response = LMSQueryResponse(
            answer="",
            course_tag="CS101",
            query_text="What is ML?",
            cache_hit=False,
            similarity=None,
            latency_ms=5.0,
            error="Milvus connection timed out",
        )

        with patch("src.api.routes.query.process_lms_query", return_value=error_response):
            resp = client.post(
                "/api/v1/query",
                json={"query": "What is ML?", "course_tag": "CS101"},
            )

        assert resp.status_code == 500

    def test_custom_similarity_threshold_is_forwarded(self, client, mock_kb_manager):
        """Verify the threshold from the request body reaches process_lms_query."""
        mock_kb_manager.partition_exists.return_value = True

        mock_response = LMSQueryResponse(
            answer="Some answer.",
            course_tag="CS101",
            query_text="What is ML?",
            cache_hit=False,
            similarity=None,
            latency_ms=50.0,
        )

        with patch("src.api.routes.query.process_lms_query", return_value=mock_response) as mock_fn:
            client.post(
                "/api/v1/query",
                json={"query": "What is ML?", "course_tag": "CS101", "similarity_threshold": 0.85},
            )
            _, kwargs = mock_fn.call_args
            assert kwargs.get("similarity_threshold") == pytest.approx(0.85)


# ─────────────────────────────────────────────────────────────────────────────
# Courses endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestCourses:

    def test_list_courses_returns_200(self, client):
        with patch("src.api.routes.courses.get_milvus_client") as mock_client:
            mock_client.return_value.list_partitions.return_value = [
                "_default", "CS101", "MATH202", "ENG301"
            ]
            resp = client.get("/api/v1/courses")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3   # _default is excluded
        tags = [c["course_tag"] for c in body["courses"]]
        assert "CS101"   in tags
        assert "MATH202" in tags
        assert "ENG301"  in tags
        assert "_default" not in tags

    def test_list_courses_excludes_internal_partitions(self, client):
        with patch("src.api.routes.courses.get_milvus_client") as mock_client:
            mock_client.return_value.list_partitions.return_value = ["_default"]
            resp = client.get("/api/v1/courses")

        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_create_new_course_returns_201(self, client, mock_kb_manager):
        mock_kb_manager.partition_exists.return_value = False

        with patch("src.api.routes.courses.create_kb_partition") as mock_create:
            resp = client.post("/api/v1/courses", json={"course_tag": "phys404"})

        assert resp.status_code == 201
        body = resp.json()
        assert body["created"] is True
        assert body["course_tag"] == "PHYS404"   # normalised to uppercase
        mock_create.assert_called_once_with("PHYS404")

    def test_create_existing_course_returns_200_not_201(self, client, mock_kb_manager):
        mock_kb_manager.partition_exists.return_value = True

        with patch("src.api.routes.courses.create_kb_partition") as mock_create:
            resp = client.post("/api/v1/courses", json={"course_tag": "CS101"})

        assert resp.status_code == 200
        assert resp.json()["created"] is False
        mock_create.assert_not_called()   # idempotent — no duplicate partition

    def test_create_course_invalid_tag_returns_422(self, client):
        resp = client.post("/api/v1/courses", json={"course_tag": "CS 101; DROP TABLE"})
        assert resp.status_code == 422

    def test_create_course_missing_tag_returns_422(self, client):
        resp = client.post("/api/v1/courses", json={})
        assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Cache invalidation endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheInvalidation:

    def test_invalidate_returns_200_with_key_count(self, client):
        mock_redis = MagicMock()
        # Simulate SCAN returning 3 keys then cursor=0 (done)
        mock_redis.scan.return_value = (0, ["k1", "k2", "k3"])

        with patch("src.api.routes.cache.get_redis_client", return_value=mock_redis):
            resp = client.delete("/api/v1/cache/CS101")

        assert resp.status_code == 200
        body = resp.json()
        assert body["keys_deleted"] == 3
        assert body["course_tag"] == "CS101"
        assert "3" in body["message"]

    def test_invalidate_empty_cache_returns_200_with_zero(self, client):
        mock_redis = MagicMock()
        mock_redis.scan.return_value = (0, [])

        with patch("src.api.routes.cache.get_redis_client", return_value=mock_redis):
            resp = client.delete("/api/v1/cache/CS101")

        assert resp.status_code == 200
        assert resp.json()["keys_deleted"] == 0

    def test_invalidate_normalises_course_tag_to_uppercase(self, client):
        mock_redis = MagicMock()
        mock_redis.scan.return_value = (0, [])

        with patch("src.api.routes.cache.get_redis_client", return_value=mock_redis):
            resp = client.delete("/api/v1/cache/cs101")

        assert resp.status_code == 200
        assert resp.json()["course_tag"] == "CS101"

    def test_invalidate_invalid_course_tag_returns_400(self, client):
        resp = client.delete("/api/v1/cache/bad tag!")
        assert resp.status_code == 400

    def test_invalidate_scans_correct_pattern(self, client):
        """Verify the Redis SCAN uses the correct key pattern for the course."""
        mock_redis = MagicMock()
        mock_redis.scan.return_value = (0, [])

        with patch("src.api.routes.cache.get_redis_client", return_value=mock_redis):
            client.delete("/api/v1/cache/MATH202")

        call_kwargs = mock_redis.scan.call_args[1]
        assert call_kwargs["match"] == "lms_cache:MATH202:*"


# ─────────────────────────────────────────────────────────────────────────────
# Response contract tests (field presence)
# ─────────────────────────────────────────────────────────────────────────────

class TestResponseContracts:

    def test_query_response_always_has_request_id(self, client, mock_kb_manager):
        mock_kb_manager.partition_exists.return_value = True

        mock_response = LMSQueryResponse(
            answer="Some answer.",
            course_tag="CS101",
            query_text="What is ML?",
            cache_hit=False,
            similarity=None,
            latency_ms=50.0,
        )

        with patch("src.api.routes.query.process_lms_query", return_value=mock_response):
            resp = client.post(
                "/api/v1/query",
                json={"query": "What is ML?", "course_tag": "CS101"},
            )

        body = resp.json()
        required_fields = {"request_id", "answer", "course_tag", "cache_hit", "latency_ms"}
        assert required_fields.issubset(body.keys()), \
            f"Missing fields: {required_fields - body.keys()}"

    def test_courses_response_always_has_request_id(self, client):
        with patch("src.api.routes.courses.get_milvus_client") as mock_client:
            mock_client.return_value.list_partitions.return_value = []
            resp = client.get("/api/v1/courses")

        assert "request_id" in resp.json()

    def test_invalidate_response_always_has_request_id(self, client):
        mock_redis = MagicMock()
        mock_redis.scan.return_value = (0, [])

        with patch("src.api.routes.cache.get_redis_client", return_value=mock_redis):
            resp = client.delete("/api/v1/cache/CS101")

        assert "request_id" in resp.json()