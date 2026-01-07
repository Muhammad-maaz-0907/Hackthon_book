import os
import logging
from typing import List, Optional, Dict, Any
from vector_client import QdrantClient
from web_scraper import WebScraper
from url_chunker import URLChunker

logger = logging.getLogger(__name__)

class URLIngestor:
    def __init__(self, qdrant_client: QdrantClient, web_scraper: Optional[WebScraper] = None):
        """
        Initialize the URL ingestor with required clients

        Args:
            qdrant_client: Qdrant client for vector storage
            web_scraper: Web scraper (optional, will create default if not provided)
        """
        self.qdrant_client = qdrant_client
        self.web_scraper = web_scraper or WebScraper()
        self.url_chunker = URLChunker(qdrant_client, self.web_scraper)

    def ingest_url(self, url: str, max_chunk_size: int = 1000, overlap_size: int = 200) -> Dict[str, Any]:
        """
        Ingest a single URL into the vector store

        Args:
            url: URL to ingest
            max_chunk_size: Maximum size of each chunk
            overlap_size: Size of overlap between chunks

        Returns:
            Dictionary with ingestion results
        """
        try:
            logger.info(f"Starting ingestion for URL: {url}")

            # Chunk the URL content
            chunks = self.url_chunker.chunk_url_content(
                url=url,
                max_chunk_size=max_chunk_size,
                overlap_size=overlap_size
            )

            if not chunks:
                logger.error(f"No content extracted from URL: {url}")
                return {
                    "success": False,
                    "url": url,
                    "chunks_processed": 0,
                    "error": "No content extracted from URL"
                }

            # Add chunks to Qdrant
            self.qdrant_client.add_document_chunks(chunks)

            logger.info(f"Successfully ingested {len(chunks)} chunks from URL: {url}")

            return {
                "success": True,
                "url": url,
                "chunks_processed": len(chunks),
                "chunks": [chunk.chunk_id for chunk in chunks]
            }

        except Exception as e:
            logger.error(f"Error ingesting URL {url}: {str(e)}")
            return {
                "success": False,
                "url": url,
                "chunks_processed": 0,
                "error": str(e)
            }

    def ingest_urls(self, urls: List[str], max_chunk_size: int = 1000, overlap_size: int = 200) -> List[Dict[str, Any]]:
        """
        Ingest multiple URLs into the vector store

        Args:
            urls: List of URLs to ingest
            max_chunk_size: Maximum size of each chunk
            overlap_size: Size of overlap between chunks

        Returns:
            List of ingestion results for each URL
        """
        results = []

        for url in urls:
            result = self.ingest_url(url, max_chunk_size, overlap_size)
            results.append(result)

            # Add a small delay to avoid overwhelming the target websites
            import time
            time.sleep(0.1)

        return results

    def is_url_processed(self, url: str) -> bool:
        """
        Check if a URL has already been processed by looking for similar content in the vector store

        Args:
            url: URL to check

        Returns:
            True if URL has been processed, False otherwise
        """
        # For now, we'll implement a basic check by searching for the URL in the metadata
        # In a more advanced implementation, we could check for content similarity
        try:
            # This is a basic implementation - in practice, you might want to search for
            # similar content or check a database of processed URLs
            domain = url.split('/')[2] if '//' in url else url.split('/')[0]
            search_results = self.qdrant_client.search(f"domain:{domain}", limit=1)

            # If we find results from the same domain with similar content, consider it processed
            return len(search_results) > 0
        except Exception:
            return False

if __name__ == "__main__":
    # Example usage
    from qdrant_client import QdrantClient

    # Initialize clients
    qdrant_client = QdrantClient()
    web_scraper = WebScraper()
    url_ingestor = URLIngestor(qdrant_client, web_scraper)

    # Example: Ingest a single URL
    result = url_ingestor.ingest_url("https://docs.ros.org/en/humble/")
    print(f"Ingestion result: {result}")