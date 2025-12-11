#!/usr/bin/env python3
"""
Test script to verify RAG functionality with sample book content
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import from app
sys.path.insert(0, str(Path(__file__).parent))

from app.services.rag import get_rag_service
from app.services.qdrant import get_qdrant_service

def test_rag_functionality():
    """
    Test the RAG functionality with sample queries
    """
    print("Testing RAG functionality with sample book content...")

    # Get the RAG service
    rag_service = get_rag_service()
    qdrant_service = get_qdrant_service()

    print("\n1. Testing document retrieval...")
    # Test retrieving relevant chunks
    query = "What are the markdown features?"
    print(f"Query: {query}")

    try:
        # Retrieve relevant chunks
        chunks = rag_service.retrieve_relevant_chunks(query)
        print(f"Retrieved {len(chunks)} relevant chunks")

        if chunks:
            print("First chunk preview:")
            print(f"Content: {chunks[0]['content'][:200]}...")
            print(f"Metadata: {chunks[0]['metadata']}")
        else:
            print("No chunks found - this might indicate an issue with ingestion or retrieval")

    except Exception as e:
        print(f"Error retrieving chunks: {str(e)}")

    print("\n2. Testing response generation...")
    # Test generating a response
    try:
        response = rag_service.generate_response(query, chunks, user_id=None)
        print(f"Generated response: {response[:200]}...")
    except Exception as e:
        print(f"Error generating response: {str(e)}")

    print("\n3. Testing full query pipeline...")
    # Test the full query pipeline
    try:
        result = rag_service.query(query, selected_text=None, user_id=None)
        print(f"Full query result: {result}")
    except Exception as e:
        print(f"Error in full query: {str(e)}")

    print("\n4. Testing with selected text (priority mode)...")
    # Test with selected text to trigger priority mode
    selected_text = "Markdown features in Docusaurus include custom components and MDX."
    try:
        result_with_selection = rag_service.query(query, selected_text=selected_text, user_id=None)
        print(f"Query with selected text result: {result_with_selection}")
    except Exception as e:
        print(f"Error in query with selection: {str(e)}")

    print("\n5. Testing collection statistics...")
    # Check collection stats
    try:
        collection_info = qdrant_service.client.get_collection("physical_ai_book")
        print(f"Collection vectors count: {collection_info.points_count}")
    except Exception as e:
        print(f"Error getting collection info: {str(e)}")

    print("\nRAG functionality test completed!")

if __name__ == "__main__":
    test_rag_functionality()