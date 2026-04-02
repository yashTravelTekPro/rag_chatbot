import json
import os
import threading
from typing import Dict
from app.data.models import BotStats

class StatsStore:
    """
    Simple JSON-based storage for bot statistics
    using file locking to handle concurrent writes
    """
    
    def __init__(self, stats_file: str = "data/stats.json"):
        self.stats_file = stats_file
        self.lock = threading.Lock()  # prevent race conditions
        
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)        
        if not os.path.exists(stats_file):
            with open(stats_file, 'w') as f:
                json.dump({}, f)
    
    def record_chat(self, bot_id: str, latency_ms: float, tokens_used: int, was_unanswered: bool) -> None:
        """
        Record metrics for a single chat interaction
        """
        with self.lock:  
            stats = self._load_stats()
            if bot_id not in stats:
                stats[bot_id] = {
                    "total_messages": 0,
                    "total_latency_ms": 0.0,
                    "total_tokens": 0,
                    "unanswered_questions": 0
                }
            
            stats[bot_id]["total_messages"] += 1
            stats[bot_id]["total_latency_ms"] += latency_ms
            stats[bot_id]["total_tokens"] += tokens_used
            
            if was_unanswered:
                stats[bot_id]["unanswered_questions"] += 1
            
            self._save_stats(stats)
    
    def get_stats(self, bot_id: str) -> BotStats:
        """
        Get aggregated statistics for a bot
        """
        with self.lock:
            stats = self._load_stats()            
            if bot_id not in stats:
                return BotStats(
                    bot_id=bot_id,
                    total_messages=0,
                    average_latency_ms=0.0,
                    estimated_cost_usd=0.0,
                    unanswered_questions=0
                )
            
            bot_data = stats[bot_id]
            
            total_msgs = bot_data["total_messages"]
            avg_latency = bot_data["total_latency_ms"] / total_msgs if total_msgs > 0 else 0.0
            
            cost_per_1k_tokens = 0.0001
            estimated_cost = (bot_data["total_tokens"] / 1000.0) * cost_per_1k_tokens
            
            return BotStats(
                bot_id=bot_id,
                total_messages=total_msgs,
                average_latency_ms=round(avg_latency, 2),
                estimated_cost_usd=round(estimated_cost, 6),
                unanswered_questions=bot_data["unanswered_questions"]
            )
    
    def _load_stats(self) -> Dict:
        """
        Load stats from JSON file
        """
        try:
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_stats(self, stats: Dict) -> None:
        """
        Save stats to JSON file
        """
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
