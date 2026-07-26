import logging
import threading
from typing import Optional

from pymilvus import MilvusClient, MilvusException

from config.settings import settings

logger = logging.getLogger(__name__)


class MilvusConnectionManager:

    _instance: Optional["MilvusConnectionManager"] = None
    _lock: threading.Lock = threading.Lock()
    _client: Optional[MilvusClient] = None

    def __new__(cls) -> "MilvusConnectionManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:   
                    cls._instance = super().__new__(cls)
        return cls._instance

    def connect(self) -> None:
        if self._client is not None:
            return

        logger.info("Connecting to Milvus at %s …", settings.milvus_uri)
        try:
            kwargs: dict = {
                "uri": settings.milvus_uri,
                "db_name": settings.milvus_db_name,
            }
            if settings.milvus_token:
                kwargs["token"] = settings.milvus_token

            self._client = MilvusClient(**kwargs)
            logger.info("Milvus connection established.")
        except MilvusException as exc:
            logger.exception("Failed to connect to Milvus: %s", exc)
            raise

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            self.connect()
        return self._client  

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Milvus connection closed.")


_manager = MilvusConnectionManager()


def get_milvus_client() -> MilvusClient:
    return _manager.client