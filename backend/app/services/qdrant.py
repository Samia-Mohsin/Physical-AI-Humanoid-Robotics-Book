from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentChunk(BaseModel):
    """Model for document chunks to be stored in Qdrant"""
    id: str
    content: str
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None

class QdrantService:
    """Service class for Qdrant vector database operations"""

    def __init__(self):
        """Initialize Qdrant client with environment configuration"""
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")

        # Initialize Qdrant client
        if self.qdrant_url and "localhost" in self.qdrant_url:
            # For local development, use in-memory Qdrant
            logger.info("Using in-memory Qdrant for local development")
            self.client = QdrantClient(":memory:")
        elif self.qdrant_api_key:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=10.0
            )
        else:
            # For local development without API key
            self.client = QdrantClient(
                url=self.qdrant_url,
                timeout=10.0
            )

        # Initialize OpenAI client for embeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
        else:
            logger.warning("OPENAI_API_KEY not set. Embedding functionality will be limited.")
            self.openai_client = None

        self.collection_name = "physical_ai_book"
        logger.info(f"Qdrant client initialized with URL: {self.qdrant_url}")

    def create_collection(self, vector_size: int = 1536) -> bool:
        """
        Create a collection in Qdrant for storing document chunks
        Default vector size is 1536 (for text-embedding-3-large OpenAI embeddings)
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return True

            # Create new collection with HNSW index for efficient similarity search
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                ),
                # Enable payload indexing for metadata search
                optimizers_config=models.OptimizersConfigDiff(
                    memmap_threshold=20000,
                    indexing_threshold=20000
                )
            )

            logger.info(f"Collection '{self.collection_name}' created successfully")
            return True

        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            return False

    def upsert_documents(self, chunks: List[DocumentChunk]) -> bool:
        """
        Upsert (insert/update) document chunks to Qdrant collection
        """
        try:
            points = []
            for chunk in chunks:
                # Generate embedding if vector is not provided
                if chunk.vector is None:
                    embedding = self._generate_embedding(chunk.content)
                else:
                    embedding = chunk.vector

                # Create a PointStruct for each chunk
                point = models.PointStruct(
                    id=chunk.id,
                    vector=embedding,
                    payload={
                        "content": chunk.content,
                        "metadata": chunk.metadata
                    }
                )
                points.append(point)

            # Upsert the points to the collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True  # Wait for operation to complete
            )

            logger.info(f"Upserted {len(chunks)} document chunks to Qdrant")
            return True

        except Exception as e:
            logger.error(f"Error upserting documents: {str(e)}")
            return False

    def search_documents(self, query_vector: List[float], limit: int = 10,
                        filters: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Search for similar documents in Qdrant collection
        """
        try:
            # Build filter conditions if provided
            qdrant_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        conditions.append(
                            models.FieldCondition(
                                key=f"metadata.{key}",
                                match=models.MatchValue(value=value)
                            )
                        )
                    elif isinstance(value, list):
                        conditions.append(
                            models.FieldCondition(
                                key=f"metadata.{key}",
                                match=models.MatchAny(any=value)
                            )
                        )

                if conditions:
                    qdrant_filter = models.Filter(must=conditions)

            # Perform the search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            # Convert results to DocumentChunk objects
            results = []
            for result in search_results:
                if result.payload:
                    chunk = DocumentChunk(
                        id=result.id,
                        content=result.payload.get("content", ""),
                        metadata=result.payload.get("metadata", {})
                    )
                    results.append(chunk)

            logger.info(f"Found {len(results)} similar documents")
            return results

        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

    def search_documents_by_text(self, query: str, limit: int = 10,
                                filters: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Search for documents by text query (generates embedding internally)
        """
        try:
            # Generate embedding for the query
            query_embedding = self._generate_embedding(query)

            # Call the search method with the generated embedding
            return self.search_documents(query_embedding, limit, filters)

        except Exception as e:
            logger.error(f"Error searching documents by text: {str(e)}")
            return []

    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents from Qdrant collection by IDs
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=ids
                ),
                wait=True
            )

            logger.info(f"Deleted {len(ids)} documents from Qdrant")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False

    def get_document_by_id(self, doc_id: str) -> Optional[DocumentChunk]:
        """
        Retrieve a specific document by its ID
        """
        try:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id],
                with_payload=True,
                with_vectors=False
            )

            if records and len(records) > 0:
                record = records[0]
                if record.payload:
                    return DocumentChunk(
                        id=record.id,
                        content=record.payload.get("content", ""),
                        metadata=record.payload.get("metadata", {})
                    )

            return None

        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {str(e)}")
            return None

    def get_all_documents(self, limit: int = 100) -> List[DocumentChunk]:
        """
        Retrieve all documents from the collection (useful for debugging/management)
        """
        try:
            records = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            results = []
            for record in records[0]:  # records returns (records, next_page_offset)
                if record.payload:
                    chunk = DocumentChunk(
                        id=record.id,
                        content=record.payload.get("content", ""),
                        metadata=record.payload.get("metadata", {})
                    )
                    results.append(chunk)

            return results

        except Exception as e:
            logger.error(f"Error retrieving all documents: {str(e)}")
            return []

    def count_documents(self) -> int:
        """
        Count total number of documents in the collection
        """
        try:
            count = self.client.count(
                collection_name=self.collection_name
            )
            return count.count
        except Exception as e:
            logger.error(f"Error counting documents: {str(e)}")
            return 0

    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection (use with caution!)
        """
        try:
            # Get all document IDs
            all_records = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on expected size
                with_payload=False,
                with_vectors=False
            )

            if all_records[0]:  # records return (records, next_page_offset)
                ids = [record.id for record in all_records[0]]
                if ids:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=models.PointIdsList(points=ids),
                        wait=True
                    )
                    logger.info(f"Cleared {len(ids)} documents from collection")

            return True

        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI
        """
        if not self.openai_client:
            # Fallback: return a simple mock embedding for testing purposes
            # In production, this should not happen as API key is required
            logger.warning("OpenAI client not available, returning mock embedding")
            # Return a mock embedding vector of 1536 dimensions (standard for text-embedding-3-large)
            import numpy as np
            # Create a deterministic mock embedding based on the text content
            text_hash = hash(text) % (2**32)
            np.random.seed(text_hash)
            return np.random.uniform(-0.1, 0.1, 1536).tolist()

        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-large"  # Using text-embedding-3-large as per spec
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return mock embedding as fallback
            import numpy as np
            text_hash = hash(text) % (2**32)
            np.random.seed(text_hash)
            return np.random.uniform(-0.1, 0.1, 1536).tolist()

# Global instance of QdrantService
qdrant_service = QdrantService()

def get_qdrant_service() -> QdrantService:
    """Get the global instance of QdrantService"""
    return qdrant_service