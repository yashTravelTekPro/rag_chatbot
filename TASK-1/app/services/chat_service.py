import time
from typing import AsyncGenerator, List, Dict, Optional
from app.services.embedding_service import EmbeddingService
from app.services.llm_client import LLMClient
from app.data.vector_store import VectorStore
from app.data.stats_store import StatsStore

class ChatService:
    """
    Handles chat interactions with the bot
    keeping the bot honest - only answer from the knowledge base
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        llm_client: LLMClient,
        stats_store: StatsStore
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_client = llm_client
        self.stats_store = stats_store
    
    async def process_chat(
        self,
        bot_id: str,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Process a chat request and stream the response
        """
        start_time = time.time()
        
        if not self.vector_store.bot_exists(bot_id):
            yield "Error: Bot not found"
            return
        
        query_embedding = self.embedding_service.generate_single_embedding(user_message)
        relevant_chunks = self.vector_store.search(bot_id, query_embedding, top_k=5)        
        system_prompt = self._build_system_prompt(relevant_chunks)
        full_response = ""
        async for chunk in self.llm_client.generate_streaming(
            system_prompt=system_prompt,
            user_message=user_message,
            conversation_history=conversation_history
        ):
            full_response += chunk
            yield chunk
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        total_text = system_prompt + user_message + full_response
        if conversation_history:
            for msg in conversation_history:
                total_text += msg.get("content", "")
        tokens_used = self.llm_client.estimate_tokens(total_text)
        was_unanswered = self._is_unanswered_response(full_response)
        
        self.stats_store.record_chat(bot_id, latency_ms, tokens_used, was_unanswered)
    
    def _build_system_prompt(self, chunks: List) -> str:
        """
        Construct system prompt with grounding instructions
        """
        context = ""
        for i, chunk in enumerate(chunks, 1):
            context += f"--- Chunk {i} ---\n{chunk.text}\n\n"
        
        prompt = f"""
You are a helpful AI assistant.

CONTEXT:
{context}

USER QUESTION:
{{question}}

INSTRUCTIONS:
- Answer the question using ONLY the provided context.
- Provide a clear, complete, and well-explained answer (not too short).
- Combine information from multiple parts of the context if needed.
- DO NOT mention chunk numbers, sources, or phrases like "according to the context".
- DO NOT explicitly say where the information came from.
- Write the answer naturally, as if you already know it.
- If the answer is not present in the context, respond with:
  "I don't have enough information to answer that."

STYLE:
- Keep the answer concise but sufficiently detailed.
- Use simple and clear language.
- Avoid unnecessary repetition.

ANSWER:
"""
    
        return prompt
    
    def _is_unanswered_response(self, response: str) -> bool:
        """
        Check if the bot indicated it couldn't answer
        """
        unanswered_phrases = [
            "don't have enough information",
            "not in my knowledge base",
            "cannot answer",
            "can't answer",
            "don't know",
            "no information",
            "not mentioned in the context"
        ]
        
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in unanswered_phrases)
