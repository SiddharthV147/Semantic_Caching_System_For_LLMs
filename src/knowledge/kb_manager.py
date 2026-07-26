import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from pymilvus import MilvusException

from config.constants import *
from src.database.milvus_client import *
from src.embeddings.embedding_service import *

logger = logging.getLogger(__name__)

_PARTITION_NOT_FOUND_CODE = 800  

# Course Not Found Exception
class CourseNotFoundError(Exception):

    def __init__(self, course_tag: str) -> None:
        self.course_tag = course_tag
        super().__init__(f"Course '{course_tag}' not found. ")



@dataclass(frozen=True)
class KBChunk:
    text:       str
    score:      float
    course_tag: str
    metadata:   dict = field(default_factory=dict)

    def as_context_string(self) -> str:
        source = self.metadata.get("source", "unknown source")
        return f"[{source}] {self.text}"


@dataclass
class KBResult:
    chunks:     list[KBChunk]
    course_tag: str
    query_text: str

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def as_context_block(self) -> str:
        if self.is_empty:
            return "No relevant course material found."
        parts = [f"{i+1}. {chunk.as_context_string()}" for i, chunk in enumerate(self.chunks)]
        return "\n\n".join(parts)


class KBManager:

    def __init__(self) -> None:
        self._milvus   = get_milvus_client()
        self._embedder = get_embedder()

    def query_kb(
        self,
        query_text: str,
        course_tag: str,
        top_k: int = TOP_K_KB ) -> KBResult:
        
        query_vector = self._embedder.embed(query_text)

        try:
            results = self._milvus.search(
                collection_name=KB_COLLECTION_NAME,
                data=[query_vector],
                anns_field=FIELD_CHUNK_VECTOR,
                search_params=KB_SEARCH_PARAMS,
                limit=top_k,
                partition_names=[course_tag],           
                output_fields=[FIELD_CHUNK_TEXT, FIELD_METADATA, FIELD_COURSE_TAG],
            )
        except MilvusException as exc:
            if _is_partition_not_found(exc, course_tag):
                raise CourseNotFoundError(course_tag) from exc
            logger.error("Milvus KB search failed | course=%s | error=%s", course_tag, exc)
            raise

        raw_hits = results[0] if results else []

        chunks: list[KBChunk] = []
        for hit in raw_hits:
            entity   = hit.get("entity", {})
            raw_meta = entity.get(FIELD_METADATA, "{}")
            try:
                metadata = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
            except json.JSONDecodeError:
                metadata = {"raw": raw_meta}

            chunks.append(KBChunk(
                text=entity.get(FIELD_CHUNK_TEXT, ""),
                score=hit.get("distance", 0.0),
                course_tag=entity.get(FIELD_COURSE_TAG, course_tag),
                metadata=metadata,
            ))

        logger.info(
            "KB query | course=%s | top_k=%d | returned=%d | query='%.60s'",
            course_tag, top_k, len(chunks), query_text,
        )
        return KBResult(chunks=chunks, course_tag=course_tag, query_text=query_text)

    def partition_exists(self, course_tag: str) -> bool:
        """
        Quick check — does a partition for this course exist?
        Use this in the orchestrator to give early, clear errors.
        """
        try:
            partitions = set(self._milvus.list_partitions(KB_COLLECTION_NAME))
            return course_tag in partitions
        except MilvusException as exc:
            logger.warning("Could not list KB partitions: %s", exc)
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_partition_not_found(exc: MilvusException, course_tag: str) -> bool:
    """
    Heuristic detection for PARTITION_NOT_FOUND errors.
    pymilvus does not always set exc.code reliably across versions,
    so we also inspect the message string as a fallback.
    """
    code_match    = getattr(exc, "code", None) == _PARTITION_NOT_FOUND_CODE
    message_match = course_tag in str(exc) and "partition" in str(exc).lower()
    return code_match or message_match


# ── Module-level singleton ────────────────────────────────────────────────────
_kb_manager: Optional[KBManager] = None


def get_kb_manager() -> KBManager:
    global _kb_manager
    if _kb_manager is None:
        _kb_manager = KBManager()
    return _kb_manager