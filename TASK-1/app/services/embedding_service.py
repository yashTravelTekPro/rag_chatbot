from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingService:
    """
    Generates embeddings using sentence-transformers
    using a lightweight model for demo purposes
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Convert text chunks into vector embeddings
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        return embeddings
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text (used for queries)
        """
        embedding = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        
        norm = np.linalg.norm(embedding)
        embedding = embedding / norm
        
        return embedding
