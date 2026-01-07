"""
Example script demonstrating the URL ingestion pipeline
"""

import asyncio
import os
from vector_client import QdrantClient
from web_scraper import WebScraper
from url_ingestor import URLIngestor

async def main():
    """
    Example usage of the URL ingestion pipeline
    """
    print("Initializing URL ingestion pipeline...")

    # Initialize clients
    # Note: Make sure to set your environment variables before running
    qdrant_client = QdrantClient()
    web_scraper = WebScraper()
    url_ingestor = URLIngestor(qdrant_client, web_scraper)

    # Example URLs to ingest
    urls = [
        "https://docs.ros.org/en/humble/",
        "https://github.com/features/copilot",
    ]

    print(f"Starting ingestion for {len(urls)} URLs...")

    for url in urls:
        print(f"\nProcessing URL: {url}")

        # Ingest the URL
        result = url_ingestor.ingest_url(
            url=url,
            max_chunk_size=1000,
            overlap_size=200
        )

        if result["success"]:
            print(f"✓ Successfully processed {result['chunks_processed']} chunks")
        else:
            print(f"✗ Failed to process URL: {result['error']}")

    print("\nIngestion completed!")

if __name__ == "__main__":
    # Check if required environment variables are set
    required_vars = ["QDRANT_URL", "QDRANT_API_KEY"]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if "COHERE_API_KEY" not in os.environ:
        print("Warning: COHERE_API_KEY not set, will use SentenceTransformer fallback")

    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these environment variables before running the script.")
        exit(1)

    asyncio.run(main())