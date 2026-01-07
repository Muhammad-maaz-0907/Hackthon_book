import asyncio
import os
from dotenv import load_dotenv
from document_chunker import DocumentChunker
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

async def main():
    """Main function to index documents"""
    print("Starting document indexing process...")

    # Initialize clients
    qdrant_client = QdrantClient()
    chunker = DocumentChunker(qdrant_client)

    # Index documents from the docs directory
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
    total_chunks = chunker.index_documents(docs_dir)

    print(f"Successfully indexed {total_chunks} chunks from the documentation!")

if __name__ == "__main__":
    asyncio.run(main())