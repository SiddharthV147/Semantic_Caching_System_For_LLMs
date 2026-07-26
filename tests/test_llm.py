import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from types import SimpleNamespace

from src.llm.llm_service import (
    BaseLLM, MockLLM, HuggingFaceAPILLM, LLMFactory, get_llm
)
from src.orchestrator import (
    process_lms_query, LMSQueryResponse, _build_prompt, settings_llm_model
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_hf_response(content: str):
    """Build a fake InferenceClient chat_completion response."""
    message = SimpleNamespace(content=content)
    choice  = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _hf_llm_with_mock_client(response_text: str) -> HuggingFaceAPILLM:
    """
    Construct a HuggingFaceAPILLM with InferenceClient fully mocked.
    The HF token check is bypassed via patched settings.
    """
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = _make_hf_response(response_text)

    with patch("src.llm.llm_service.settings") as mock_settings, \
         patch("src.llm.llm_service.InferenceClient", return_value=mock_client):

        mock_settings.hf_token        = "hf_test_token_fake"
        mock_settings.llm_provider    = "auto"
        mock_settings.llm_model_name  = "Qwen/Qwen2.5-7B-Instruct"
        mock_settings.llm_max_new_tokens = 512
        mock_settings.llm_temperature = 0.3
        mock_settings.llm_timeout     = 60
        mock_settings.llm_backend     = "huggingface_api"

        llm = HuggingFaceAPILLM.__new__(HuggingFaceAPILLM)
        llm._client      = mock_client
        llm._model       = "Qwen/Qwen2.5-7B-Instruct"
        llm._max_tokens  = 512
        llm._temperature = 0.3
        llm._lock        = __import__("threading").Lock()

    return llm, mock_client


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_llm_singleton():
    LLMFactory.reset()
    yield
    LLMFactory.reset()


@pytest.fixture
def cache_miss(request):
    """CacheManager returning a CacheMiss by default."""
    from src.cache.cache_manager import CacheMiss
    m = MagicMock()
    m.search_cache.return_value = CacheMiss(
        query_text="What is backpropagation?",
        course_tag="CS101",
        best_score=0.0,
    )
    m.update_cache.return_value = "lms_cache:CS101:abc123"
    return m


@pytest.fixture
def empty_kb():
    """KBManager returning empty results."""
    from src.knowledge.kb_manager import KBResult
    m = MagicMock()
    m.partition_exists.return_value = True
    m.query_kb.return_value = KBResult(
        chunks=[], course_tag="CS101", query_text="What is backpropagation?"
    )
    return m


@pytest.fixture
def kb_with_chunks():
    """KBManager returning 2 realistic chunks."""
    from src.knowledge.kb_manager import KBResult, KBChunk
    m = MagicMock()
    m.partition_exists.return_value = True
    m.query_kb.return_value = KBResult(
        chunks=[
            KBChunk(
                text="Backpropagation computes gradients using the chain rule.",
                score=0.91, course_tag="CS101",
                metadata={"source": "lecture_3.pdf", "page": 12},
            ),
            KBChunk(
                text="It propagates error signals from output to input layers.",
                score=0.87, course_tag="CS101",
                metadata={"source": "lecture_3.pdf", "page": 13},
            ),
        ],
        course_tag="CS101", query_text="What is backpropagation?"
    )
    return m


# ─────────────────────────────────────────────────────────────────────────────
# BaseLLM interface
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseLLMInterface:

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            BaseLLM()

    def test_subclass_without_generate_fails(self):
        class Incomplete(BaseLLM):
            pass
        with pytest.raises(TypeError):
            Incomplete()

    def test_valid_subclass_works(self):
        class Minimal(BaseLLM):
            def generate(self, prompt: str) -> str:
                return "ok"
        assert Minimal().generate("anything") == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# MockLLM
# ─────────────────────────────────────────────────────────────────────────────

class TestMockLLM:

    def test_returns_string(self):
        assert isinstance(MockLLM().generate("prompt"), str)

    def test_fixed_response_returned_verbatim(self):
        assert MockLLM(fixed_response="fixed").generate("anything") == "fixed"

    def test_extracts_question_from_prompt(self):
        prompt = "Course material:\nSome text\n\nQuestion: What is entropy?\n\nAnswer:"
        result = MockLLM().generate(prompt)
        assert "entropy" in result

    def test_same_prompt_gives_same_result(self):
        llm = MockLLM()
        p = "Question: What is ML?\n\nAnswer:"
        assert llm.generate(p) == llm.generate(p)

    def test_empty_prompt_does_not_raise(self):
        result = MockLLM().generate("")
        assert isinstance(result, str)

    def test_no_fixed_response_mentions_mock(self):
        result = MockLLM().generate("Question: test?\n\nAnswer:")
        assert "MOCK" in result.upper()


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFaceAPILLM — mocked HTTP
# ─────────────────────────────────────────────────────────────────────────────

class TestHuggingFaceAPILLM:

    def test_generate_returns_model_content(self):
        llm, mock_client = _hf_llm_with_mock_client(
            "Backpropagation uses the chain rule of calculus."
        )
        result = llm.generate("Question: What is backpropagation?")
        assert result == "Backpropagation uses the chain rule of calculus."

    def test_generate_strips_whitespace(self):
        llm, _ = _hf_llm_with_mock_client("  Answer with spaces.   ")
        assert llm.generate("prompt") == "Answer with spaces."

    def test_empty_response_returns_fallback(self):
        llm, _ = _hf_llm_with_mock_client("   ")
        result = llm.generate("prompt")
        assert "could not generate" in result.lower()

    def test_chat_completion_called_with_system_and_user_messages(self):
        llm, mock_client = _hf_llm_with_mock_client("Good answer.")
        llm.generate("What is ML?")

        call_args = mock_client.chat_completion.call_args
        messages  = call_args[1]["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is ML?"

    def test_correct_model_name_sent_to_api(self):
        llm, mock_client = _hf_llm_with_mock_client("Answer.")
        llm.generate("prompt")
        assert mock_client.chat_completion.call_args[1]["model"] == "Qwen/Qwen2.5-7B-Instruct"

    def test_rate_limit_error_returns_friendly_message(self):
        llm, mock_client = _hf_llm_with_mock_client("irrelevant")
        mock_client.chat_completion.side_effect = Exception("429 rate limit exceeded")
        result = llm.generate("prompt")
        assert "rate-limited" in result.lower()

    def test_401_auth_error_returns_friendly_message(self):
        llm, mock_client = _hf_llm_with_mock_client("irrelevant")
        mock_client.chat_completion.side_effect = Exception("401 unauthorized invalid token")
        result = llm.generate("prompt")
        assert "authentication" in result.lower() or "hf_token" in result.lower()

    def test_503_model_loading_returns_friendly_message(self):
        llm, mock_client = _hf_llm_with_mock_client("irrelevant")
        mock_client.chat_completion.side_effect = Exception("503 model is loading")
        result = llm.generate("prompt")
        assert "warming up" in result.lower() or "retry" in result.lower()

    def test_generic_error_returns_error_string_not_raise(self):
        llm, mock_client = _hf_llm_with_mock_client("irrelevant")
        mock_client.chat_completion.side_effect = RuntimeError("network timeout")
        result = llm.generate("prompt")
        assert "LLM API error" in result
        assert "network timeout" in result

    def test_missing_token_raises_value_error(self):
        with patch("src.llm.llm_service.settings") as mock_settings, \
             patch("src.llm.llm_service.InferenceClient"):
            mock_settings.hf_token     = ""
            mock_settings.llm_provider = "auto"
            mock_settings.llm_timeout  = 60
            with pytest.raises(ValueError, match="HF_TOKEN"):
                HuggingFaceAPILLM()

    def test_generate_is_thread_safe(self):
        import threading
        llm, mock_client = _hf_llm_with_mock_client("Thread-safe answer.")
        results, errors = [], []

        def worker():
            try:
                results.append(llm.generate("concurrent prompt"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(errors) == 0
        assert len(results) == 5
        assert all(r == "Thread-safe answer." for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# LLMFactory
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMFactory:

    def test_returns_mock_when_backend_is_mock(self):
        with patch("src.llm.llm_service.settings") as s:
            s.llm_backend = "mock"
            assert isinstance(LLMFactory.get(), MockLLM)

    def test_singleton_same_instance(self):
        with patch("src.llm.llm_service.settings") as s:
            s.llm_backend = "mock"
            a = LLMFactory.get()
            b = LLMFactory.get()
        assert a is b

    def test_reset_forces_new_instance(self):
        with patch("src.llm.llm_service.settings") as s:
            s.llm_backend = "mock"
            a = LLMFactory.get()
        LLMFactory.reset()
        with patch("src.llm.llm_service.settings") as s:
            s.llm_backend = "mock"
            b = LLMFactory.get()
        assert a is not b

    def test_invalid_backend_raises(self):
        with patch("src.llm.llm_service.settings") as s:
            s.llm_backend = "unknown_backend"
            with pytest.raises(ValueError, match="Unknown LLM_BACKEND"):
                LLMFactory.get()

    def test_get_llm_returns_base_llm(self):
        with patch("src.llm.llm_service.settings") as s:
            s.llm_backend = "mock"
            assert isinstance(get_llm(), BaseLLM)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPrompt:

    def test_contains_query(self):
        p = _build_prompt("What is entropy?", "Entropy measures uncertainty.", "CS101")
        assert "What is entropy?" in p

    def test_contains_context(self):
        p = _build_prompt("Q?", "Entropy measures uncertainty.", "CS101")
        assert "Entropy measures uncertainty." in p

    def test_contains_course_tag(self):
        p = _build_prompt("Q?", "Some context.", "INFO202")
        assert "INFO202" in p

    def test_no_context_uses_general_knowledge_branch(self):
        p = _build_prompt("Q?", "No relevant course material found.", "CS101")
        assert "general knowledge" in p.lower()

    def test_empty_context_uses_general_knowledge_branch(self):
        p = _build_prompt("Q?", "", "CS101")
        assert "general knowledge" in p.lower()

    def test_ends_with_answer_marker(self):
        p = _build_prompt("Q?", "Some context.", "CS101")
        assert p.strip().endswith("Answer:")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestrator:

    def test_cache_miss_calls_llm_and_stores_answer(self, cache_miss, empty_kb):
        llm = MockLLM(fixed_response="Backpropagation uses chain rule.")

        with patch("src.orchestrator.get_cache_manager", return_value=cache_miss), \
             patch("src.orchestrator.get_kb_manager",    return_value=empty_kb):
            result = process_lms_query("What is backpropagation?", "CS101", llm=llm)

        assert result.ok
        assert result.cache_hit is False
        assert result.answer == "Backpropagation uses chain rule."
        cache_miss.update_cache.assert_called_once_with(
            query_text="What is backpropagation?",
            response_text="Backpropagation uses chain rule.",
            course_tag="CS101",
        )

    def test_cache_hit_skips_llm_and_kb(self, cache_miss, empty_kb):
        from src.cache.cache_manager import CacheHit
        cache_miss.search_cache.return_value = CacheHit(
            redis_key="k", course_tag="CS101",
            response="Cached answer.", similarity=0.97,
            query_text="What is backpropagation?",
        )
        llm = MockLLM(fixed_response="Should NOT appear.")

        with patch("src.orchestrator.get_cache_manager", return_value=cache_miss), \
             patch("src.orchestrator.get_kb_manager",    return_value=empty_kb):
            result = process_lms_query("What is backpropagation?", "CS101", llm=llm)

        assert result.cache_hit is True
        assert result.answer == "Cached answer."
        assert result.similarity == pytest.approx(0.97)
        empty_kb.query_kb.assert_not_called()
        cache_miss.update_cache.assert_not_called()

    def test_kb_chunks_included_in_prompt(self, cache_miss, kb_with_chunks):
        captured = []

        class CapturingLLM(BaseLLM):
            def generate(self, prompt: str) -> str:
                captured.append(prompt)
                return "Answer from context."

        with patch("src.orchestrator.get_cache_manager", return_value=cache_miss), \
             patch("src.orchestrator.get_kb_manager",    return_value=kb_with_chunks):
            result = process_lms_query("What is backpropagation?", "CS101", llm=CapturingLLM())

        assert result.kb_chunks_used == 2
        assert "chain rule" in captured[0]
        assert "error signals" in captured[0]

    def test_course_not_found_returns_error_response(self, cache_miss):
        from src.knowledge.kb_manager import CourseNotFoundError
        kb = MagicMock()
        kb.query_kb.side_effect = CourseNotFoundError("UNKNOWN")

        with patch("src.orchestrator.get_cache_manager", return_value=cache_miss), \
             patch("src.orchestrator.get_kb_manager",    return_value=kb):
            result = process_lms_query("Q?", "UNKNOWN", llm=MockLLM())

        assert not result.ok
        assert "UNKNOWN" in result.error
        assert result.answer == ""

    def test_llm_exception_returns_error_not_raise(self, cache_miss, empty_kb):
        class ExplodingLLM(BaseLLM):
            def generate(self, prompt: str) -> str:
                raise RuntimeError("GPU on fire")

        with patch("src.orchestrator.get_cache_manager", return_value=cache_miss), \
             patch("src.orchestrator.get_kb_manager",    return_value=empty_kb):
            result = process_lms_query("Q?", "CS101", llm=ExplodingLLM())

        assert not result.ok
        assert "LLM error" in result.error
        cache_miss.update_cache.assert_not_called()

    def test_cache_update_failure_is_non_fatal(self, cache_miss, empty_kb):
        cache_miss.update_cache.side_effect = Exception("Redis down")
        llm = MockLLM(fixed_response="Good answer.")

        with patch("src.orchestrator.get_cache_manager", return_value=cache_miss), \
             patch("src.orchestrator.get_kb_manager",    return_value=empty_kb):
            result = process_lms_query("Q?", "CS101", llm=llm)

        assert result.ok
        assert result.answer == "Good answer."

    def test_llm_model_populated_on_miss(self, cache_miss, empty_kb):
        with patch("src.orchestrator.get_cache_manager", return_value=cache_miss), \
             patch("src.orchestrator.get_kb_manager",    return_value=empty_kb), \
             patch("src.orchestrator.settings_llm_model", return_value="Qwen/Qwen2.5-7B-Instruct"):
            result = process_lms_query("Q?", "CS101", llm=MockLLM())

        assert result.llm_model == "Qwen/Qwen2.5-7B-Instruct"

    def test_llm_model_empty_on_cache_hit(self, cache_miss, empty_kb):
        from src.cache.cache_manager import CacheHit
        cache_miss.search_cache.return_value = CacheHit(
            redis_key="k", course_tag="CS101",
            response="Cached.", similarity=0.99, query_text="Q?"
        )
        with patch("src.orchestrator.get_cache_manager", return_value=cache_miss), \
             patch("src.orchestrator.get_kb_manager",    return_value=empty_kb):
            result = process_lms_query("Q?", "CS101", llm=MockLLM())

        assert result.llm_model == ""

    def test_response_has_correct_fields(self, cache_miss, empty_kb):
        with patch("src.orchestrator.get_cache_manager", return_value=cache_miss), \
             patch("src.orchestrator.get_kb_manager",    return_value=empty_kb):
            result = process_lms_query("Q?", "CS101", llm=MockLLM(fixed_response="A."))

        assert isinstance(result, LMSQueryResponse)
        assert result.course_tag == "CS101"
        assert result.query_text == "Q?"
        assert isinstance(result.latency_ms, float)
        assert result.latency_ms >= 0

    def test_hf_api_rate_limit_message_stored_in_cache(self, cache_miss, empty_kb):
        """Even an API error message (rate limit) gets stored — it's the LLM's response."""
        llm, mock_client = _hf_llm_with_mock_client("irrelevant")
        mock_client.chat_completion.side_effect = Exception("429 rate limit exceeded")

        with patch("src.orchestrator.get_cache_manager", return_value=cache_miss), \
             patch("src.orchestrator.get_kb_manager",    return_value=empty_kb):
            result = process_lms_query("Q?", "CS101", llm=llm)

        # Rate limit returns a user-friendly string, not an exception
        assert result.ok
        assert "rate" in result.answer.lower() or "limit" in result.answer.lower()