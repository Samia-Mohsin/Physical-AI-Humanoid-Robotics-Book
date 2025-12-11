#!/usr/bin/env python3
"""
Simple test script to verify RAG functionality with sample book content
This version bypasses the database dependency to focus on core RAG functionality
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import from app
sys.path.insert(0, str(Path(__file__).parent))

from app.services.qdrant import get_qdrant_service

def test_qdrant_functionality():
    """
    Test the Qdrant functionality directly
    """
    print("Testing Qdrant functionality with sample book content...")

    # Get the Qdrant service
    qdrant_service = get_qdrant_service()

    print("\n1. Testing collection information...")
    try:
        collection_info = qdrant_service.client.get_collection("physical_ai_book")
        print(f"Collection vectors count: {collection_info.points_count}")
        print(f"Collection name: {collection_info.config.params.vectors_config.size if collection_info.config.params.vectors_config else 'N/A'}")
    except Exception as e:
        print(f"Error getting collection info: {str(e)}")

    print("\n2. Testing search functionality...")
    # Create a mock query vector for testing search (using the same mock embedding approach)
    mock_query = "What are the markdown features?"

    # Generate a mock embedding for the query (using the same approach as in qdrant.py)
    import numpy as np
    text_hash = hash(mock_query) % (2**32)
    np.random.seed(text_hash)
    mock_embedding = np.random.uniform(-0.1, 0.1, 1536).tolist()

    try:
        # Search for similar documents
        search_results = qdrant_service.client.search(
            collection_name="physical_ai_book",
            query_vector=mock_embedding,
            limit=5  # Get top 5 results
        )

        print(f"Found {len(search_results)} search results")

        if search_results:
            print("\nFirst result preview:")
            first_result = search_results[0]
            print(f"ID: {first_result.id}")
            print(f"Score: {first_result.score}")
            print(f"Content preview: {first_result.payload.get('content', '')[:200]}...")
            print(f"Metadata: {first_result.payload.get('metadata', {})}")
        else:
            print("No search results found - this might indicate an issue with ingestion")

    except Exception as e:
        print(f"Error searching documents: {str(e)}")

    print("\n3. Testing document listing...")
    try:
        # List all points in the collection
        points = qdrant_service.client.scroll(
            collection_name="physical_ai_book",
            limit=10
        )

        print(f"Retrieved {len(points[0])} documents from collection")

        if points[0]:
            for i, point in enumerate(points[0][:3]):  # Show first 3
                print(f"\nDocument {i+1}:")
                print(f"  ID: {point.id}")
                print(f"  Content preview: {point.payload.get('content', '')[:100]}...")
                print(f"  Metadata: {point.payload.get('metadata', {})}")

    except Exception as e:
        print(f"Error listing documents: {str(e)}")

    print("\nQdrant functionality test completed!")

if __name__ == "__main__":
    test_qdrant_functionality()