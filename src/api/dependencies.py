from fastapi import Request

from src.cache.cache_manager import CacheManager
from src.knowledge.kb_manager import KBManager


def get_cache_manager(request: Request) -> CacheManager:
    """Retrieve the CacheManager stored in app.state during lifespan startup."""
    return request.app.state.cache_manager


def get_kb_manager(request: Request) -> KBManager:
    """Retrieve the KBManager stored in app.state during lifespan startup."""
    return request.app.state.kb_manager