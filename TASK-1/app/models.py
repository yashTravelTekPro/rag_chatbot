from pydantic import BaseModel, field_validator
from typing import Optional, List, Dict

class UploadRequest(BaseModel):
    content: Optional[str] = None
    url: Optional[str] = None

    @field_validator('content', 'url')
    @classmethod
    def check_exactly_one(cls, v, info):
        values = info.data
        content = values.get('content')
        url = values.get('url')
        if info.field_name == 'url':
            if (content and url) or (not content and not url):
                raise ValueError("Exactly one of 'content' or 'url' must be provided")
        return v


class ChatRequest(BaseModel):
    bot_id: str
    user_message: str
    conversation_history: Optional[List[Dict[str, str]]] = None

    @field_validator('user_message')
    @classmethod
    def message_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("user_message cannot be empty")
        return v


class UploadResponse(BaseModel):
    bot_id: str
    chunks_created: int
    message: str


class StatsResponse(BaseModel):
    bot_id: str
    total_messages: int
    average_latency_ms: float
    estimated_cost_usd: float
    unanswered_questions: int


class ErrorResponse(BaseModel):
    error: str
    detail: str
