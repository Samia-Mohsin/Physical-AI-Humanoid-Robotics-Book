from typing import List, Dict, Any, Optional
from openai import OpenAI
from .qdrant import get_qdrant_service, DocumentChunk as QdrantDocumentChunk
from .neon import get_neon_service
from dotenv import load_dotenv
import os
import logging
from datetime import datetime

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    """Service class for Retrieval-Augmented Generation functionality"""

    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant_service = get_qdrant_service()
        self.neon_service = get_neon_service()

    def retrieve_relevant_chunks(self, query: str, selected_text: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks from Qdrant
        If selected_text is provided, prioritize chunks from that text
        """
        try:
            if selected_text and selected_text.strip():
                # When selected text is provided, search specifically for chunks related to that text
                logger.info(f"Searching for chunks related to selected text: {selected_text[:100]}...")
                results = self.qdrant_service.search_documents_by_text(selected_text, limit=limit*2)

                # If we get results from selected text, return those
                if results:
                    return [
                        {
                            "content": chunk.content,
                            "metadata": chunk.metadata,
                            "score": 1.0  # High relevance since it's the selected text
                        }
                        for chunk in results
                    ][:limit]
                else:
                    # If no specific results from selected text, fall back to general query
                    results = self.qdrant_service.search_documents_by_text(query, limit=limit)
            else:
                # General query without selected text
                results = self.qdrant_service.search_documents_by_text(query, limit=limit)

            # Convert results to the expected format
            return [
                {
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "score": 1.0  # Placeholder score
                }
                for chunk in results
            ]

        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {str(e)}")
            return []

    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]], user_id: Optional[str] = None) -> str:
        """
        Generate response using OpenAI with retrieved context
        """
        try:
            # Get user profile if available for personalization
            user_profile = None
            if user_id:
                user_profile = self.neon_service.get_user_profile(user_id)

            # Prepare context from retrieved chunks
            context = "\n\n".join([chunk["content"] for chunk in context_chunks])

            # Prepare the prompt for the LLM
            if context:
                system_prompt = f"""
                You are an expert assistant for the Physical AI & Humanoid Robotics educational platform.
                Answer the user's question based on the provided context from the book content.

                {f"User profile: {user_profile}" if user_profile else ""}

                Context:
                {context}

                Instructions:
                - Provide accurate answers based only on the context provided
                - If the answer is not in the context, say so clearly
                - Include relevant citations to the source material
                - Use technical terminology appropriately but explain complex concepts when needed
                - Keep responses educational and focused on Physical AI & Humanoid Robotics
                """
            else:
                system_prompt = """
                You are an expert assistant for the Physical AI & Humanoid Robotics educational platform.
                The user's question doesn't match any specific book content, but you can still provide general information
                about Physical AI & Humanoid Robotics based on your knowledge.
                """

            # Call OpenAI API to generate response
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Using gpt-4o-mini as specified in the requirements
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=1000,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I encountered an error while processing your request. Please try again."

    def query(self, query: str, selected_text: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main query method that orchestrates the RAG process
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")

            # Retrieve relevant chunks
            context_chunks = self.retrieve_relevant_chunks(query, selected_text)

            # Generate response
            response = self.generate_response(query, context_chunks, user_id)

            # Prepare result with sources
            result = {
                "response": response,
                "sources": [
                    {
                        "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                        "metadata": chunk["metadata"]
                    }
                    for chunk in context_chunks
                ],
                "selected_text_used": bool(selected_text)
            }

            logger.info("Query processed successfully")
            return result

        except Exception as e:
            logger.error(f"Error in query method: {str(e)}")
            return {
                "response": "I encountered an error while processing your request. Please try again.",
                "sources": [],
                "selected_text_used": bool(selected_text)
            }

# Global instance of RAGService
rag_service = RAGService()

def get_rag_service() -> RAGService:
    """Get the global instance of RAGService"""
    return rag_service