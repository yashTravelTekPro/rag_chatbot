try:
    import nltk
except Exception:
    nltk = None
import os
import re
from typing import List
from pathlib import Path
from app.data.models import Chunk

if nltk:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _NLTK_DATA_DIR = _PROJECT_ROOT / "nltk_data"
    os.makedirs(_NLTK_DATA_DIR, exist_ok=True   )
    nltk.data.path.insert(0, str(_NLTK_DATA_DIR))
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', download_dir=str(_NLTK_DATA_DIR), quiet=True)
        except Exception:
            pass


class ChunkingEngine:
    """
    Splits text into semantic chunks with overlap
    overlap helps preserve context at boundaries
    """
    
    def __init__(self, target_size: int = 800, overlap: int = 50):
        self.target_size = target_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, source: str = "text") -> List[Chunk]:
        """
        Break text into overlapping chunks at sentence boundaries
        """
        if not text or not text.strip():
            return []
        
        if nltk:
            try:
                sentences = nltk.sent_tokenize(text)
            except Exception:
                sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        else:
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        chunks = []
        current_chunk = []
        current_size = 0
        position = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_size + sentence_len > self.target_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    position=position,
                    source=source,
                    bot_id=None  # will be set later
                ))
                position += 1
                overlap_sentences = []
                overlap_size = 100
                
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_len
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                position=position,
                source=source,
                bot_id=None
            ))
        
        return chunks
