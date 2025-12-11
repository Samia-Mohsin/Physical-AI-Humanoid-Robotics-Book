from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from ..services.rag import get_rag_service
import logging
import json
from fastapi.responses import StreamingResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/query",
    tags=["query"],
    responses={404: {"description": "Not found"}},
)

# Request models
class QueryRequest(BaseModel):
    query: str
    selected_text: Optional[str] = None
    user_id: Optional[str] = None

# Response models
class QueryResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    selected_text_used: bool

def event_stream(query: str, selected_text: Optional[str], user_id: Optional[str]):
    """
    Generator function for streaming responses
    """
    try:
        # Get the RAG service instance
        rag_service = get_rag_service()

        # Get relevant chunks first
        context_chunks = rag_service.retrieve_relevant_chunks(query, selected_text)

        # Prepare context from retrieved chunks
        context = "\n\n".join([chunk["content"] for chunk in context_chunks])

        # Prepare the prompt for the LLM
        system_prompt = f"""
        You are an expert assistant for the Physical AI & Humanoid Robotics educational platform.
        Answer the user's question based on the provided context from the book content.

        Context:
        {context}

        Instructions:
        - Provide accurate answers based only on the context provided
        - If the answer is not in the context, say so clearly
        - Include relevant citations to the source material
        - Use technical terminology appropriately but explain complex concepts when needed
        - Keep responses educational and focused on Physical AI & Humanoid Robotics
        """

        # For streaming, we'll simulate streaming by breaking the response into chunks
        # In a real implementation, you would use OpenAI's streaming API
        full_response = rag_service.generate_response(query, context_chunks, user_id)

        # Send sources first
        sources_data = [
            {
                "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                "metadata": chunk["metadata"]
            }
            for chunk in context_chunks
        ]

        yield f"data: {json.dumps({'type': 'sources', 'sources': sources_data})}\n\n"

        # Stream the response content in chunks
        words = full_response.split()
        for i in range(0, len(words), 5):  # Send 5 words at a time
            chunk = ' '.join(words[i:i+5])
            yield f"data: {json.dumps({'type': 'content', 'content': chunk + ' '})}\n\n"

        # Send end marker
        yield f"data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error in streaming: {str(e)}")
        yield f"data: {json.dumps({'type': 'error', 'content': 'An error occurred while processing your query'})}\n\n"

@router.post("/", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint for the RAG chatbot
    Accepts a query, optional selected text, and optional user ID
    Returns a response with sources and whether selected text was used
    """
    try:
        logger.info(f"Received query: {request.query[:100]}...")
        if request.selected_text:
            logger.info(f"Selected text provided: {request.selected_text[:100]}...")

        # Get the RAG service instance
        rag_service = get_rag_service()

        # Process the query
        result = rag_service.query(
            query=request.query,
            selected_text=request.selected_text,
            user_id=request.user_id
        )

        logger.info("Query processed successfully")
        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your query"
        )

@router.post("/stream")
async def query_stream_endpoint(request: QueryRequest):
    """
    Streaming query endpoint for the RAG chatbot
    Returns a streaming response with Server-Sent Events
    """
    try:
        logger.info(f"Received streaming query: {request.query[:100]}...")
        if request.selected_text:
            logger.info(f"Selected text provided: {request.selected_text[:100]}...")

        return StreamingResponse(
            event_stream(request.query, request.selected_text, request.user_id),
            media_type="text/plain"
        )

    except Exception as e:
        logger.error(f"Error processing streaming query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your query"
        )

# Additional endpoint for health check of the query service
@router.get("/health")
async def query_health():
    """
    Health check endpoint for the query service
    """
    return {"status": "healthy", "service": "RAG Query Service"}