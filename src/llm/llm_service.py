import logging
import threading
from abc import ABC, abstractmethod
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class BaseLLM(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate an answer for the given prompt.
        Must never raise — return an error string instead of propagating exceptions.
        """


class HuggingFaceAPILLM(BaseLLM):

    def __init__(self) -> None:
        from huggingface_hub import InferenceClient

        token = settings.hf_token
        if not token:
            raise ValueError(
                "HF_TOKEN is not set. "
            )

        provider = settings.llm_provider if settings.llm_provider != "auto" else None

        self._client   = InferenceClient(
            api_key=token,
            provider=provider,
            timeout=settings.llm_timeout,
        )
        self._model    = settings.llm_model_name
        self._max_tokens = settings.llm_max_new_tokens
        self._temperature = settings.llm_temperature
        self._lock     = threading.Lock()

        logger.info(
            "HuggingFaceAPILLM ready | model=%s | provider=%s",
            self._model,
            settings.llm_provider,
        )

    def generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful, concise course assistant. "
                    "Answer only based on the provided course material. "
                    "If the answer is not in the material, say so clearly."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        try:
            with self._lock:
                completion = self._client.chat_completion(
                    messages=messages,
                    model=self._model,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )

            answer = completion.choices[0].message.content.strip()

            if not answer:
                return "I could not generate an answer based on the provided course material."

            logger.debug(
                "LLM response | model=%s | %d tokens generated",
                self._model, len(answer.split()),
            )
            return answer

        except Exception as exc:
            logger.exception("HuggingFace API call failed.")
            err = str(exc).lower()
            if "rate limit" in err or "429" in err:
                return (
                    "Service is temporarily rate-limited. "
                    "Please wait a moment and try again."
                )
            if "401" in err or "unauthorized" in err or "invalid token" in err:
                return (
                    "Authentication failed. "
                    "Check that HF_TOKEN in your .env is correct and has Read permissions."
                )
            if "503" in err or "model is loading" in err:
                return (
                    "The model is warming up on the provider's servers. "
                    "Please retry in 20–30 seconds."
                )
            return f"LLM API error: {exc}"


class MockLLM(BaseLLM):

    def __init__(self, fixed_response: Optional[str] = None) -> None:
        self._fixed = fixed_response
        logger.warning("MockLLM active — NOT suitable for production.")

    def generate(self, prompt: str) -> str:
        if self._fixed is not None:
            return self._fixed

        question = ""
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith("Question:"):
                question = stripped.replace("Question:", "").strip()
                break

        return (
            f"[MOCK RESPONSE] "
            f"This is a deterministic test answer for: '{question}'. "
            f"Set LLM_BACKEND=huggingface_api and HF_TOKEN=hf_xxx for real responses."
        )

class LLMFactory:
    _instance: Optional[BaseLLM] = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def _build(cls) -> BaseLLM:
        backend = settings.llm_backend.lower()
        if backend == "mock":
            return MockLLM()
        if backend == "huggingface_api":
            return HuggingFaceAPILLM()
        raise ValueError(
            f"Unknown LLM_BACKEND='{backend}'. "
            "Valid options: 'huggingface_api', 'mock'."
        )

    @classmethod
    def get(cls) -> BaseLLM:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls._build()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None


def get_llm() -> BaseLLM:
    return LLMFactory.get()
