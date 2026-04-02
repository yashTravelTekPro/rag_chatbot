import uuid
import requests
from typing import Tuple, Optional, Any
import io
import os
from typing import List
from app.services.chunking_engine import ChunkingEngine
from app.data.vector_store import VectorStore
import re
import html

class UploadService:
    """
    Handles knowledge base uploads
    """
    
    def __init__(
        self,
        chunking_engine: ChunkingEngine,
        embedding_service: Any,
        vector_store: VectorStore
    ):
        self.chunking_engine = chunking_engine
        self.embedding_service = embedding_service
        self.vector_store = vector_store
    
    def process_upload(
        self,
        content: Optional[str] = None,
        url: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        Process uploaded content and create a bot
        returns (bot_id, chunk_count)
        """
        if url:
            text_content = self._fetch_url_content(url)
            source = url
        elif content:
            text_content = content
            source = "text"
        else:
            raise ValueError("Either content or url must be provided")
        
        if not text_content or not text_content.strip():
            raise ValueError("Content is empty or invalid")
        
        bot_id = str(uuid.uuid4())
        
        chunks = self.chunking_engine.chunk_text(text_content, source=source)
        
        if not chunks:
            raise ValueError("No chunks generated from content")
        
        for chunk in chunks:
            chunk.bot_id = bot_id
        
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_service.generate_embeddings(chunk_texts)
        
        self.vector_store.add_bot(bot_id, embeddings, chunks)
        
        return bot_id, len(chunks)
    
    def _fetch_url_content(self, url: str) -> str:
        """
        Fetch content from a URL
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '').lower()

            if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                pdf_bytes = response.content
                try:
                    from langchain.document_loaders import UnstructuredPDFLoader
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmpf:
                        tmpf.write(pdf_bytes)
                        tmp_path = tmpf.name
                    try:
                        loader = UnstructuredPDFLoader(tmp_path)
                        docs = loader.load()
                        text = "\n".join(getattr(d, 'page_content', getattr(d, 'page_content', getattr(d, 'content', ''))) for d in docs)
                        text = re.sub(r'\s+', ' ', text).strip()
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
                except Exception:
                    try:
                        from pypdf import PdfReader
                        reader = PdfReader(io.BytesIO(pdf_bytes))
                        pages: List[str] = []
                        for p in reader.pages:
                            try:
                                pages.append(p.extract_text() or "")
                            except Exception:
                                pages.append("")
                        text = "\n".join(pages)
                        text = re.sub(r'\s+', ' ', text).strip()
                    except Exception:
                        raise ValueError("PDF content detected but no suitable PDF parser available. Install 'langchain' with its unstructured dependencies or 'pypdf'.")
            else:
                text = response.text

            if 'html' in content_type or bool(re.search(r'<\s*html', text, re.I)):
                text = re.sub(r'<script[\s\S]*?</script>', ' ', text, flags=re.I)
                text = re.sub(r'<style[\s\S]*?</style>', ' ', text, flags=re.I)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = html.unescape(text)
                text = re.sub(r'\s+', ' ', text).strip()

            return text
        
        except requests.Timeout:
            raise ValueError(f"URL fetch timeout: {url}")
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch URL: {str(e)}")
