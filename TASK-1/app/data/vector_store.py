import faiss
import numpy as np
import json
import os
from typing import List, Dict
from app.data.models import Chunk

class VectorStore:
    """
    Manages FAISS indices for multiple bots
    keeping things simple with file-based storage
    """
    
    def __init__(self, data_dir: str = "data/indices"):
        self.data_dir = data_dir
        self.indices: Dict[str, faiss.Index] = {}
        self.metadata: Dict[str, List[Chunk]] = {}
        os.makedirs(data_dir, exist_ok=True)
    
    def add_bot(self, bot_id: str, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """
        Create a new FAISS index for a bot
        """
        dimension = embeddings.shape[1]        
        index = faiss.IndexFlatL2(dimension)        
        index.add(embeddings.astype('float32'))        
        self.indices[bot_id] = index
        self.metadata[bot_id] = chunks        
        self.save_to_disk(bot_id)
    
    def search(self, bot_id: str, query_embedding: np.ndarray, top_k: int = 5) -> List[Chunk]:
        """
        Find most similar chunks for a query
        """
        if bot_id not in self.indices:
            if not self.load_from_disk(bot_id):
                return []
        
        index = self.indices[bot_id]
        chunks = self.metadata[bot_id]
        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = index.search(query_embedding.astype('float32'), min(top_k, len(chunks)))
        
        results = []
        for idx in indices[0]:
            if idx < len(chunks):  # safety check
                results.append(chunks[idx])
        
        return results
    
    def bot_exists(self, bot_id: str) -> bool:
        """
        Check if we have data for this bot
        """
        if bot_id in self.indices:
            return True
        
        index_path = os.path.join(self.data_dir, f"{bot_id}.index")
        return os.path.exists(index_path)
    
    def save_to_disk(self, bot_id: str) -> None:
        """
        Persist bot's index and metadata to disk
        """
        if bot_id not in self.indices:
            return
        
        index_path = os.path.join(self.data_dir, f"{bot_id}.index")
        faiss.write_index(self.indices[bot_id], index_path)
        
        metadata_path = os.path.join(self.data_dir, f"{bot_id}_metadata.json")
        chunks_data = [
            {
                "text": chunk.text,
                "position": chunk.position,
                "source": chunk.source,
                "bot_id": chunk.bot_id
            }
            for chunk in self.metadata[bot_id]
        ]
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    def load_from_disk(self, bot_id: str) -> bool:
        """
        Load bot's index and metadata from disk
        """
        index_path = os.path.join(self.data_dir, f"{bot_id}.index")
        metadata_path = os.path.join(self.data_dir, f"{bot_id}_metadata.json")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            return False
        
        try:
            index = faiss.read_index(index_path)
            self.indices[bot_id] = index
            with open(metadata_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)            
            chunks = [
                Chunk(
                    text=c["text"],
                    position=c["position"],
                    source=c["source"],
                    bot_id=c.get("bot_id")
                )
                for c in chunks_data
            ]
            self.metadata[bot_id] = chunks
            
            return True
        except Exception as e:
            print(f"Error loading bot {bot_id}: {e}")
            return False
