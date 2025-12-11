#!/usr/bin/env python3
"""
Test the API endpoints to verify RAG functionality
"""

import subprocess
import time
import requests
import threading
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from app
sys.path.insert(0, str(Path(__file__).parent))

def start_server():
    """Start the FastAPI server"""
    import uvicorn
    from app.main import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

def test_api_functionality():
    """Test the API endpoints"""
    print("Testing API functionality...")

    # First, let's run the ingestion script to populate the database
    print("Running ingestion script...")
    result = subprocess.run([sys.executable, "ingest/load_book_to_qdrant.py"],
                          capture_output=True, text=True, cwd=".")
    print("Ingestion output:")
    print(result.stdout)
    if result.stderr:
        print("Ingestion errors:")
        print(result.stderr)

    print("\nIngestion completed. Now testing API endpoints...")

    # Test the health endpoint first
    try:
        response = requests.get("http://127.0.0.1:8000/query/health")
        print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")

    # Test the query endpoint
    try:
        query_data = {
            "query": "What are the markdown features?",
            "selected_text": None,
            "user_id": None
        }
        response = requests.post("http://127.0.0.1:8000/query/", json=query_data)
        print(f"Query endpoint: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response preview: {result.get('response', '')[:200]}...")
            print(f"Sources count: {len(result.get('sources', []))}")
        else:
            print(f"Query failed: {response.text}")
    except Exception as e:
        print(f"Query endpoint failed: {e}")

if __name__ == "__main__":
    test_api_functionality()