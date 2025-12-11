#!/usr/bin/env python3
"""
Document ingestion script for loading book content into Qdrant vector database.
This script loads book content from MDX files into Qdrant for RAG.
"""

import os
import sys
from pathlib import Path
import uuid

# Add the parent directory to the path so we can import from app
sys.path.insert(0, str(Path(__file__).parent.parent))

import glob
from typing import List, Dict, Any
from app.services.qdrant import get_qdrant_service, DocumentChunk
import markdown
import frontmatter
import re
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_mdx(file_path: str) -> str:
    """
    Extract text content from MDX file, removing code blocks and frontmatter
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Parse frontmatter if present
    try:
        post = frontmatter.load(file_path)
        content = post.content
    except:
        # If no frontmatter, use the whole content
        pass

    # Remove code blocks (triple backticks)
    lines = content.split('\n')
    result_lines = []
    in_code_block = False

    for line in lines:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        if not in_code_block:
            result_lines.append(line)

    text_content = '\n'.join(result_lines)

    # Convert markdown to plain text (remove markdown formatting)
    html = markdown.markdown(text_content)
    # Remove HTML tags to get plain text
    plain_text = re.sub('<[^<]+?>', '', html)

    return plain_text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks of specified size
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If this isn't the last chunk, try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings near the end
            chunk = text[start:end]
            last_period = chunk.rfind('.')
            last_exclamation = chunk.rfind('!')
            last_question = chunk.rfind('?')
            last_space = chunk.rfind(' ')

            # Choose the closest sentence boundary before chunk_size
            break_points = [bp for bp in [last_period, last_exclamation, last_question, last_space] if bp > chunk_size // 2]

            if break_points:
                break_point = max(break_points)
                end = start + break_point + 1
            else:
                # If no good break point found, break at chunk_size
                end = start + chunk_size

        chunks.append(text[start:end].strip())
        start = end - overlap  # Overlap for continuity

        # Ensure we don't get stuck in an infinite loop
        if start >= len(text):
            break

    # Remove empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    return chunks

def load_book_to_qdrant():
    """
    Load book content from MDX files to Qdrant vector store
    """
    logger.info("Starting book ingestion to Qdrant...")

    # Initialize Qdrant service
    qdrant_service = get_qdrant_service()

    # Create collection if it doesn't exist
    qdrant_service.create_collection()

    # Find all MDX files in the docs directory
    # First try the my-book/docs directory
    docs_paths = [
        "../../../my-book/docs/**/*.mdx",  # Path from backend/ingest/
        "../frontend/docs/**/*.mdx",      # Alternative path
        "../../frontend/docs/**/*.mdx"    # Another alternative
    ]

    mdx_files = []
    docs_path = ""

    for path in docs_paths:
        files = glob.glob(path, recursive=True)
        if files:
            mdx_files = files
            docs_path = path
            break

    if not mdx_files:
        # If no MDX files found, try for MD files as well
        for path in docs_paths:
            md_files = glob.glob(path.replace('**/*.mdx', '**/*.md'), recursive=True)
            if md_files:
                mdx_files = md_files
                docs_path = path.replace('**/*.mdx', '**/*.md')
                break

    logger.info(f"Found {len(mdx_files)} content files to process in {docs_path}")

    total_chunks = 0

    for file_path in mdx_files:
        logger.info(f"Processing {file_path}...")

        try:
            # Extract text content from MDX/MD file
            text_content = extract_text_from_mdx(file_path)

            # Chunk the text
            chunks = chunk_text(text_content)

            logger.info(f"  Split into {len(chunks)} chunks")

            # Prepare document chunks for upsert
            document_chunks = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    chunk_obj = DocumentChunk(
                        id=str(uuid.uuid4()),  # Use UUID for Qdrant compatibility
                        content=chunk,
                        metadata={
                            "source": file_path,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "file_path": file_path,
                            "type": "book_content",
                            "original_id": f"{os.path.basename(file_path)}_chunk_{i:04d}"  # Keep original ID for reference
                        }
                    )
                    document_chunks.append(chunk_obj)

            # Upsert all chunks for this file at once
            if document_chunks:
                success = qdrant_service.upsert_documents(document_chunks)
                if success:
                    total_chunks += len(document_chunks)
                    logger.info(f"  Successfully added {len(document_chunks)} chunks to Qdrant")
                else:
                    logger.error(f"  Failed to add chunks for {file_path} to Qdrant")

        except Exception as e:
            logger.error(f"  Error processing {file_path}: {str(e)}")
            continue

    logger.info(f"Successfully loaded {total_chunks} chunks to Qdrant")
    logger.info("Book ingestion completed!")

if __name__ == "__main__":
    load_book_to_qdrant()