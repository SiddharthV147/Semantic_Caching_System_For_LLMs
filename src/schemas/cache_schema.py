
import json
from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator

class CacheEntry(BaseModel):
    
    redis_key: str = Field(description="Redis key", max_length=256 )
    course_tag: str = Field(max_length=64)
    llm_response: str = Field(description="LLM response")
    query_text: str = Field()
    model_name: str = Field(default="unknown")
    created_at: datetime = Field( default_factory=lambda: datetime.now(timezone.utc) )
    hit_count: int = Field(default=0, ge=0)

    def to_redis_payload(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_redis_payload(cls, raw: str) -> "CacheEntry":
        return cls.model_validate_json(raw)