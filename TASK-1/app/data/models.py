from dataclasses import dataclass
from typing import Optional

@dataclass
class Chunk:
    """
    Represents a chunk of text from the knowledge base
    """
    text: str 
    position: int 
    source: str
    bot_id: Optional[str] = None 

@dataclass
class BotStats:
    """
    Statistics for a specific bot
    """
    bot_id: str
    total_messages: int 
    average_latency_ms: float 
    estimated_cost_usd: float 
    unanswered_questions: int 