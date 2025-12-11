#!/usr/bin/env python3
"""
Complete test script to verify RAG functionality from ingestion to querying
This version tests the complete flow in one script to ensure data persistence within the same in-memory instance
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import from app
sys.path.insert(0, str(Path(__file__).parent))

from app.services.qdrant import get_qdrant_service
from app.services.rag import get_rag_service

def test_complete_rag_flow():
    """
    Test the complete RAG flow from ingestion to querying
    """
    print("Testing complete RAG functionality with sample book content...")

    # Get services
    qdrant_service = get_qdrant_service()
    rag_service = get_rag_service()

    print("\n1. Testing collection information...")
    try:
        collection_info = qdrant_service.client.get_collection("physical_ai_book")
        print(f"Collection vectors count: {collection_info.points_count}")
    except Exception as e:
        print(f"Collection not found, this is expected if no data has been ingested in this session: {str(e)}")

    print("\n2. Testing document retrieval...")
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
            print("Let's try ingesting some sample content first...")

            # Let's manually create and add some test content
            from app.services.qdrant import DocumentChunk
            import uuid
            import os

            # Create some sample chunks
            sample_chunks = [
                DocumentChunk(
                    id=str(uuid.uuid4()),
                    content="Markdown features in Docusaurus include custom components, MDX support, and rich formatting options. You can embed React components directly in your markdown files.",
                    metadata={
                        "source": "test_sample.mdx",
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "file_path": "test_sample.mdx",
                        "type": "book_content",
                        "original_id": "test_sample.mdx_chunk_0000"
                    }
                ),
                DocumentChunk(
                    id=str(uuid.uuid4()),
                    content="Advanced markdown features include syntax highlighting, code blocks with language detection, and custom admonitions like notes, tips, and warnings.",
                    metadata={
                        "source": "test_sample.mdx",
                        "chunk_index": 1,
                        "total_chunks": 1,
                        "file_path": "test_sample.mdx",
                        "type": "book_content",
                        "original_id": "test_sample.mdx_chunk_0001"
                    }
                )
            ]

            # Add the sample chunks to Qdrant
            success = qdrant_service.upsert_documents(sample_chunks)
            if success:
                print("Successfully added sample content to Qdrant")

                # Try to retrieve again
                chunks = rag_service.retrieve_relevant_chunks(query)
                print(f"Retrieved {len(chunks)} relevant chunks after adding sample data")

                if chunks:
                    print("First chunk preview after adding sample data:")
                    print(f"Content: {chunks[0]['content'][:200]}...")
            else:
                print("Failed to add sample content")

    except Exception as e:
        print(f"Error retrieving chunks: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n3. Testing response generation...")
    # Test generating a response
    try:
        response = rag_service.generate_response(query, chunks if chunks else [], user_id=None)
        print(f"Generated response: {response[:200]}...")
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n4. Testing full query pipeline...")
    # Test the full query pipeline
    try:
        result = rag_service.query(query, selected_text=None, user_id=None)
        print(f"Full query result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        if isinstance(result, dict):
            print(f"Response preview: {result.get('response', 'N/A')[:200]}...")
            print(f"Sources count: {len(result.get('sources', []))}")
            print(f"Selected text used: {result.get('selected_text_used', 'N/A')}")
    except Exception as e:
        print(f"Error in full query: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n5. Testing with selected text (priority mode)...")
    # Test with selected text to trigger priority mode
    selected_text = "Markdown features in Docusaurus include custom components and MDX."
    try:
        result_with_selection = rag_service.query(query, selected_text=selected_text, user_id=None)
        print(f"Query with selected text result keys: {list(result_with_selection.keys()) if isinstance(result_with_selection, dict) else 'Not a dict'}")
        if isinstance(result_with_selection, dict):
            print(f"Response preview: {result_with_selection.get('response', 'N/A')[:200]}...")
            print(f"Selected text used: {result_with_selection.get('selected_text_used', 'N/A')}")
    except Exception as e:
        print(f"Error in query with selection: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n6. Testing collection statistics...")
    # Check collection stats
    try:
        collection_info = qdrant_service.client.get_collection("physical_ai_book")
        print(f"Final collection vectors count: {collection_info.points_count}")
    except Exception as e:
        print(f"Error getting collection info: {str(e)}")

    print("\nComplete RAG functionality test completed!")

if __name__ == "__main__":
    test_complete_rag_flow()