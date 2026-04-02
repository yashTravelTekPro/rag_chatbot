from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import logging
from app.models import (
    UploadRequest, UploadResponse, ChatRequest, 
    StatsResponse, ErrorResponse
)
from app.services.upload_service import UploadService
from app.services.chat_service import ChatService
from app.services.chunking_engine import ChunkingEngine
from app.services.embedding_service import EmbeddingService
from app.services.llm_client import LLMClient
from app.data.vector_store import VectorStore
from app.data.stats_store import StatsStore
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
router = APIRouter()
vector_store = VectorStore()
stats_store = StatsStore()
chunking_engine = ChunkingEngine()
embedding_service = EmbeddingService()
llm_client = LLMClient()
upload_service = UploadService(chunking_engine, embedding_service, vector_store)
chat_service = ChatService(vector_store, embedding_service, llm_client, stats_store)

@router.post("/upload", response_model=UploadResponse)
async def upload_knowledge_base(request: Request, payload: UploadRequest | None = None):
    """
    Upload text or URL to create a knowledge base
    """
    try:
        # Accept multiple content types and keep the endpoint documented with UploadRequest
        content_type = request.headers.get('content-type', '')

        content = None
        url = None

        if payload is not None:
            content = payload.content
            url = payload.url
            logger.info(f"Upload request received (UploadRequest) - content: {bool(content)}, url: {bool(url)}")
        else:
            if content_type.startswith('application/json'):
                try:
                    body = await request.json()
                    content = body.get('content')
                    url = body.get('url')
                    logger.info(f"Upload request received (json fallback) - content: {bool(content)}, url: {bool(url)}")
                except Exception:
                    raise HTTPException(status_code=422, detail="JSON decode error - ensure you send valid JSON or use text/plain for raw text")

            elif content_type.startswith('text/plain'):
                text_body = (await request.body()).decode('utf-8')
                content = text_body
                logger.info(f"Upload request received (text/plain) - content length: {len(content)}")

            else:
                raise HTTPException(status_code=415, detail="Unsupported content type. Send JSON (application/json) or plain text (text/plain). For JSON use {\"content\": \"...\"} or {\"url\": \"...\"}.")

        bot_id, chunk_count = upload_service.process_upload(
            content=content,
            url=url
        )
        
        logger.info(f"Bot created: {bot_id}, chunks: {chunk_count}")
        
        return UploadResponse(
            bot_id=bot_id,
            chunks_created=chunk_count,
            message=f"Knowledge base created successfully with {chunk_count} chunks"
        )
    
    except ValueError as e:
        logger.warning(f"Upload validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/chat")
async def chat_with_bot(request: ChatRequest):
    """
    Chat with a bot using its knowledge base
    streaming the response so users don't wait forever
    """
    try:
        logger.info(f"Chat request for bot: {request.bot_id}")        
        if not vector_store.bot_exists(request.bot_id):
            logger.warning(f"Bot not found: {request.bot_id}")
            raise HTTPException(status_code=404, detail=f"Bot not found: {request.bot_id}")
        
        async def generate():
            async for chunk in chat_service.process_chat(
                bot_id=request.bot_id,
                user_message=request.user_message,
                conversation_history=request.conversation_history
            ):
                yield chunk
        
        return StreamingResponse(generate(), media_type="text/plain")
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/stats/{bot_id}", response_model=StatsResponse)
async def get_bot_stats(bot_id: str):
    """
    Get statistics for a bot
    """
    try:
        logger.info(f"Stats request for bot: {bot_id}")
        
        if not vector_store.bot_exists(bot_id):
            logger.warning(f"Bot not found: {bot_id}")
            raise HTTPException(status_code=404, detail=f"Bot not found: {bot_id}")
        
        stats = stats_store.get_stats(bot_id)
        
        return StatsResponse(
            bot_id=stats.bot_id,
            total_messages=stats.total_messages,
            average_latency_ms=stats.average_latency_ms,
            estimated_cost_usd=stats.estimated_cost_usd,
            unanswered_questions=stats.unanswered_questions
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
