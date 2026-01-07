import asyncio
from pathlib import Path
from document_chunker import DocumentChunker, DocumentChunk
from qdrant_client import QdrantClient

def test_chunking():
    """Test the document chunking functionality"""
    print("Testing document chunking...")

    # Create a mock Qdrant client for testing (won't actually connect)
    qdrant_client = QdrantClient.__new__(QdrantClient)  # Create without calling __init__
    qdrant_client.client = None  # Mock the client
    qdrant_client.embedding_model = None  # Mock the embedding model
    qdrant_client.collection_name = "test_collection"

    # Create chunker instance
    chunker = DocumentChunker(qdrant_client)

    # Test content splitting
    test_content = """# Introduction

This is the introduction section of the document.

## Getting Started

Here's how to get started with the humanoid robotics book.

The book covers various topics including ROS 2, simulation, NVIDIA Isaac, and Vision-Language-Action systems.

### Prerequisites

Before starting, you should have:
- Basic programming knowledge
- Understanding of robotics concepts
- Familiarity with Linux command line

## Chapter 1: The Robotic Nervous System

This chapter covers ROS 2 fundamentals.

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software.

It provides services such as:
- Hardware abstraction
- Device drivers
- Libraries for implementing common functionality
- Message-passing between nodes
- Package management

The architecture is designed to be:
- Distributed
- Real-time capable
- Scalable
- Multi-platform
"""

    # Test splitting by headings
    sections = chunker._split_by_headings(test_content)
    print(f"Split content into {len(sections)} sections:")
    for i, (heading, content) in enumerate(sections):
        print(f"  Section {i+1}: {heading} ({len(content)} chars)")
        print(f"    Content preview: {content[:100]}...")
        print()

    # Test creating sub-chunks
    if sections:
        long_content = sections[0][1]  # Get the content of the first section
        sub_chunks = chunker._create_sub_chunks(long_content, max_chunk_size=100)
        print(f"Split first section into {len(sub_chunks)} sub-chunks:")
        for i, chunk in enumerate(sub_chunks):
            print(f"  Sub-chunk {i+1}: {len(chunk)} chars")
            print(f"    Preview: {chunk[:100]}...")
            print()

def test_single_document_chunking():
    """Test chunking a single document"""
    print("Testing single document chunking...")

    # Create a temporary markdown file for testing
    test_file = Path("test_doc.md")
    test_content = """# Test Document

This is a test document for the chunking functionality.

## Section 1

This section covers the basics.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

## Section 2

This section covers more advanced topics.

Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

### Subsection 2.1

This is a subsection with more details.

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.
"""

    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)

    try:
        # Create a mock Qdrant client for testing
        qdrant_client = QdrantClient.__new__(QdrantClient)
        qdrant_client.client = None
        qdrant_client.embedding_model = None
        qdrant_client.collection_name = "test_collection"

        chunker = DocumentChunker(qdrant_client)
        chunks = chunker._chunk_single_document(test_file)

        print(f"Created {len(chunks)} chunks from single document:")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}:")
            print(f"    Doc Path: {chunk.doc_path}")
            print(f"    Heading: {chunk.heading}")
            print(f"    Chunk ID: {chunk.chunk_id}")
            print(f"    Content Length: {len(chunk.content)} chars")
            print(f"    Content Preview: {chunk.content[:100]}...")
            print()

    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()

if __name__ == "__main__":
    test_chunking()
    print("\n" + "="*50 + "\n")
    test_single_document_chunking()