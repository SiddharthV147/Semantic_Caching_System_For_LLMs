import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.cache.cache_manager import get_cache_manager, CacheHit, CacheMiss
from src.knowledge.kb_manager import get_kb_manager, CourseNotFoundError, KBResult
from src.llm.llm_service import get_llm, BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class LMSQueryResponse:
    answer: str
    course_tag: str
    query_text: str
    cache_hit: bool
    similarity: Optional[float]   
    latency_ms: float
    kb_chunks_used: int  = 0         
    llm_model: str  = ""        
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None



def _build_prompt(query: str, context_block: str, course_tag: str) -> str:
    if context_block and context_block != "No relevant course material found.":
        return (
            f"You are a helpful course assistant for {course_tag}.\n"
            f"Use ONLY the course material below to answer the question.\n"
            f"If the answer is not in the material, say 'I don't have enough "
            f"information in the course material to answer this question.'\n\n"
            f"Course material:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
    else:
        return (
            f"You are a helpful course assistant for {course_tag}.\n"
            f"No specific course material was found for this question.\n"
            f"Answer as best you can based on general knowledge.\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )


def process_lms_query(
    user_query: str,
    course_tag: str,
    similarity_threshold: Optional[float] = None,
    top_k_kb: int = 5,
    llm: Optional[BaseLLM] = None, ) -> LMSQueryResponse:

    t_start = time.perf_counter()
    llm_svc = llm or get_llm()
    cache_mgr = get_cache_manager()
    kb_mgr = get_kb_manager()

    try:
        cache_result = cache_mgr.search_cache(
            query_text=user_query,
            course_tag=course_tag,
            similarity_threshold=similarity_threshold,
        )
    except Exception as exc:
        logger.exception("Cache search failed.")
        return _error_response(user_query, course_tag, t_start, str(exc))

    if isinstance(cache_result, CacheHit):
        logger.info(
            "Cache HIT | course=%s | sim=%.4f | latency=%.1fms",
            course_tag, cache_result.similarity, _elapsed_ms(t_start),
        )
        return LMSQueryResponse(
            answer=cache_result.response,
            course_tag=course_tag,
            query_text=user_query,
            cache_hit=True,
            similarity=cache_result.similarity,
            latency_ms=_elapsed_ms(t_start),
        )

    assert isinstance(cache_result, CacheMiss)
    logger.info(
        "Cache MISS (best=%.4f) → KB retrieval | course=%s",
        cache_result.best_score, course_tag,
    )

    try:
        kb_result: KBResult = kb_mgr.query_kb(
            query_text=user_query,
            course_tag=course_tag,
            top_k=top_k_kb,
        )
    except CourseNotFoundError as exc:
        logger.warning("Course not found: %s", exc)
        return _error_response(user_query, course_tag, t_start, str(exc))
    except Exception as exc:
        logger.exception("KB retrieval failed.")
        return _error_response(user_query, course_tag, t_start, str(exc))

    if kb_result.is_empty:
        logger.warning(
            "KB returned 0 chunks | course=%s | query='%.60s'",
            course_tag, user_query,
        )

    prompt = _build_prompt(
        query=user_query,
        context_block=kb_result.as_context_block(),
        course_tag=course_tag,
    )
    logger.debug("Prompt assembled | %d chars | %d KB chunks", len(prompt), len(kb_result.chunks))

    try:
        answer = llm_svc.generate(prompt)
        logger.info(
            "LLM generated answer | course=%s | %d chars | model=%s",
            course_tag, len(answer), settings_llm_model(),
        )
    except Exception as exc:
        logger.exception("LLM generation failed.")
        return _error_response(user_query, course_tag, t_start, f"LLM error: {exc}")

    try:
        cache_mgr.update_cache(
            query_text=user_query,
            response_text=answer,
            course_tag=course_tag,
        )
        logger.info("Cache updated | course=%s", course_tag)
    except Exception as exc:
        logger.error("Cache update failed (non-fatal): %s", exc)

    return LMSQueryResponse(
        answer=answer,
        course_tag=course_tag,
        query_text=user_query,
        cache_hit=False,
        similarity=cache_result.best_score or None,
        latency_ms=_elapsed_ms(t_start),
        kb_chunks_used=len(kb_result.chunks),
        llm_model=settings_llm_model(),
    )


def _elapsed_ms(t_start: float) -> float:
    return round((time.perf_counter() - t_start) * 1000, 2)


def _error_response(
    query: str,
    course_tag: str,
    t_start: float,
    message: str,  ) -> LMSQueryResponse:
    return LMSQueryResponse(
        answer="",
        course_tag=course_tag,
        query_text=query,
        cache_hit=False,
        similarity=None,
        latency_ms=_elapsed_ms(t_start),
        error=message,
    )


def settings_llm_model() -> str:
    from config.settings import settings
    return settings.llm_model_name if settings.llm_backend != "mock" else "mock"