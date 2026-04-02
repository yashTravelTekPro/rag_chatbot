from groq import Groq
from typing import AsyncGenerator, List, Dict, Optional
import os
from dotenv import load_dotenv
load_dotenv()
class LLMClient:
    """
    Wrapper for Groq API to generate responses
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        self.model = model
        self.client = Groq(api_key=self.api_key)
    
    async def generate_streaming(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream response from LLM
        yields chunks of text as they come in
        """
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=1024
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            yield f"Error generating response: {str(e)}"
    
    def estimate_tokens(self, text: str) -> int:
        """
        Rough token count estimation
        1 token ≈ 4 characters for English text
        """
        return len(text) // 4
    
    def calculate_cost(self, tokens: int) -> float:
        """
        Estimate cost based on token count
        adjust this based on actual Groq pricing
        """
        cost_per_1k = 0.0001
        return (tokens / 1000.0) * cost_per_1k
