#!/usr/bin/env python3
"""
Single-File Website URL Ingestion and Embedding Pipeline

This script implements a complete pipeline to:
1. Fetch content from website URLs
2. Extract clean textual content
3. Chunk text with overlap
4. Generate embeddings using Cohere (with fallback)
5. Store embeddings in Qdrant vector database

All functionality is contained in a single executable file.
"""

import argparse
import asyncio
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import bs4
import cohere
import dotenv
import qdrant_client
import requests
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a chunk of content extracted from a URL"""

    def __init__(self, doc_path: str, heading: str, chunk_id: str, content: str, page_number: Optional[int] = None):
        self.doc_path = doc_path
        self.heading = heading
        self.chunk_id = chunk_id
        self.content = content
        self.page_number = page_number


class ScrapedContent:
    """Internal model for scraped web content"""

    def __init__(self, url: str, title: str, domain: str, content: str, timestamp: int, status_code: int):
        self.url = url
        self.title = title
        self.domain = domain
        self.content = content
        self.timestamp = timestamp
        self.status_code = status_code


class WebScraper:
    """Handles fetching and extracting content from URLs"""

    def __init__(self, timeout: int = 30, user_agent: Optional[str] = None):
        self.timeout = timeout
        self.user_agent = user_agent or "Single-File-URL-Scraper/1.0"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

    def scrape_url(self, url: str) -> Optional[ScrapedContent]:
        """Scrape content from a given URL"""
        try:
            # Validate URL format
            parsed_url = urlparse(url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                logger.error(f"Invalid URL format: {url}")
                return None

            # Make request to the URL
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            # Parse HTML content
            soup = bs4.BeautifulSoup(response.content, 'html.parser')

            # Extract content
            title = self._extract_title(soup)
            text_content = self._extract_text_content(soup)
            domain = parsed_url.netloc

            # Clean up content
            cleaned_content = self._clean_content(text_content)

            return ScrapedContent(
                url=url,
                title=title,
                domain=domain,
                content=cleaned_content,
                timestamp=int(time.time()),
                status_code=response.status_code
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error scraping URL {url}: {str(e)}")
            return None

    def _extract_title(self, soup: bs4.BeautifulSoup) -> str:
        """Extract title from the HTML"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()

        # Try to find h1 as title if no title tag
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()

        return "Untitled"

    def _extract_text_content(self, soup: bs4.BeautifulSoup) -> str:
        """Extract main text content from HTML, removing navigation and ads"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()

        # Look for main content containers (common patterns)
        main_content = None
        for selector in ['main', 'article', '[role="main"]', '.content', '#content', '.post', '.article']:
            main_content = soup.select_one(selector)
            if main_content:
                break

        # If no main content container found, use the body
        if not main_content:
            main_content = soup.find('body')

        # If still no content, use the full soup
        if not main_content:
            main_content = soup

        # Extract text, preserving paragraph structure
        paragraphs = []
        for element in main_content.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td']):
            text = element.get_text(separator=' ', strip=True)
            if text:
                paragraphs.append(text)

        return '\n\n'.join(paragraphs)

    def _clean_content(self, content: str) -> str:
        """Clean extracted content by removing extra whitespace and common artifacts"""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)

        # Remove common artifacts (e.g., multiple dots, special characters)
        content = re.sub(r'[.]{3,}', '...', content)
        content = content.strip()

        return content


class CohereClient:
    """Handles embedding generation using Cohere with fallback to SentenceTransformer"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key is required. Set COHERE_API_KEY environment variable.")

        # Initialize Cohere client
        self.client = cohere.Client(self.api_key)

        # Initialize fallback embedding model
        self.fallback_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Cohere embedding model
        self.embedding_model = "embed-english-v3.0"

    def embed_text(self, text: str, input_type: str = "search_document") -> List[float]:
        """Generate embedding for text using Cohere"""
        try:
            response = self.client.embed(
                texts=[text],
                model=self.embedding_model,
                input_type=input_type
            )
            return response.embeddings[0]
        except Exception as e:
            logger.warning(f"Cohere embedding failed: {str(e)}, falling back to SentenceTransformer")
            # Fallback to SentenceTransformer
            embedding = self.fallback_model.encode(text)
            return embedding.tolist()

    def embed_texts(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        """Generate embeddings for multiple texts using Cohere"""
        try:
            # Cohere has a limit of 96 texts per request
            batch_size = 96
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embed(
                    texts=batch,
                    model=self.embedding_model,
                    input_type=input_type
                )
                all_embeddings.extend(response.embeddings)

            return all_embeddings
        except Exception as e:
            logger.warning(f"Cohere batch embedding failed: {str(e)}, falling back to SentenceTransformer")
            # Fallback to SentenceTransformer
            embeddings = []
            for text in texts:
                embedding = self.fallback_model.encode(text)
                embeddings.append(embedding.tolist())
            return embeddings


class QdrantClient:
    """Handles vector storage in Qdrant"""

    def __init__(self):
        # Initialize Qdrant client
        self.client = qdrant_client.QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=True
        )

        # Determine vector size based on embedding model (using Cohere by default)
        self.vector_size = 1024  # Cohere embeddings are 1024-dimensional

        # Collection name for the content
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "website_content")

        # Create collection if it doesn't exist
        self._create_collection()

    def _create_collection(self):
        """Create the Qdrant collection if it doesn't exist"""
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
        except:
            # Create collection with vector configuration
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,  # Cohere embedding size
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created collection '{self.collection_name}' with vector size {self.vector_size}")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using Cohere client"""
        # Create Cohere client for embedding
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable is required for embedding")

        cohere_client = CohereClient(cohere_api_key)
        return cohere_client.embed_text(text)

    def add_document_chunks(self, chunks: List[DocumentChunk]):
        """Add document chunks to the vector store"""
        points = []

        for chunk in chunks:
            # Generate embedding for the chunk content
            vector = self.embed_text(chunk.content)

            # Create payload with metadata
            payload = {
                "doc_path": chunk.doc_path,
                "heading": chunk.heading,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "page_number": chunk.page_number
            }

            # Create point for Qdrant
            point = models.PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload=payload
            )

            points.append(point)

        # Upload points to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        logger.info(f"Added {len(points)} chunks to Qdrant collection")

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for relevant chunks based on query"""
        # Create Cohere client for embedding
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable is required for embedding")

        cohere_client = CohereClient(cohere_api_key)
        query_vector = cohere_client.embed_text(query)

        # Perform search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )

        # Extract results with metadata
        results = []
        for hit in search_results:
            result = {
                "doc_path": hit.payload["doc_path"],
                "heading": hit.payload["heading"],
                "chunk_id": hit.payload["chunk_id"],
                "content": hit.payload["content"],
                "page_number": hit.payload.get("page_number"),
                "score": hit.score
            }
            results.append(result)

        return results


def validate_url(url: str) -> bool:
    """Validate if a URL is properly formatted"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def normalize_text(text: str) -> str:
    """Normalize whitespace and encoding in text"""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters if needed
    return text.strip()


def chunk_text(content: str, max_chunk_size: int = 1000, overlap_size: int = 200) -> List[str]:
    """Split text into chunks with overlap"""
    if len(content) <= max_chunk_size:
        return [content]

    # Split content into sentences
    sentences = re.split(r'[.!?]+\s+', content)
    chunks = []
    current_chunk = ""
    current_chunk_size = 0

    for sentence in sentences:
        sentence_size = len(sentence)

        # If adding this sentence would exceed the max size
        if current_chunk_size + sentence_size > max_chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # For overlapping chunks, start new chunk with overlap from previous chunk
            if overlap_size > 0 and chunks:
                # Get the end portion of the previous chunk as overlap
                prev_chunk = chunks[-1]
                overlap_start_idx = max(0, len(prev_chunk) - overlap_size)
                current_chunk = prev_chunk[overlap_start_idx:]
                current_chunk_size = len(current_chunk)
            else:
                current_chunk = ""
                current_chunk_size = 0

        # Add sentence to current chunk
        if current_chunk:
            current_chunk += " " + sentence
        else:
            current_chunk = sentence
        current_chunk_size += sentence_size + 1  # +1 for the space

    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Handle any remaining chunks that are still too large by splitting by paragraphs
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            # Split large chunk by paragraphs
            paragraph_chunks = split_by_paragraphs(chunk, max_chunk_size, overlap_size)
            final_chunks.extend(paragraph_chunks)

    return final_chunks


def split_by_paragraphs(text: str, max_chunk_size: int, overlap_size: int) -> List[str]:
    """Split large text by paragraphs with overlap"""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    current_chunk_size = 0

    for paragraph in paragraphs:
        paragraph_size = len(paragraph)

        if current_chunk_size + paragraph_size > max_chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Add overlap from previous chunk
            if overlap_size > 0 and chunks:
                prev_chunk = chunks[-1]
                overlap_start_idx = max(0, len(prev_chunk) - overlap_size)
                current_chunk = prev_chunk[overlap_start_idx:]
                current_chunk_size = len(current_chunk)
            else:
                current_chunk = ""
                current_chunk_size = 0

        if current_chunk:
            current_chunk += "\n\n" + paragraph
        else:
            current_chunk = paragraph
        current_chunk_size += paragraph_size + 2  # +2 for the \n\n

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def generate_chunk_id(doc_path: str, index: int) -> str:
    """Generate a stable chunk identifier"""
    return f"{doc_path.replace('/', '_').replace('.', '_')}_{index}_{int(time.time())}"


def process_urls(urls: List[str], max_chunk_size: int = 1000, overlap_size: int = 200) -> Dict:
    """Main function to process a list of URLs"""
    logger.info(f"Starting to process {len(urls)} URLs")

    web_scraper = WebScraper()
    qdrant_client = QdrantClient()

    total_chunks = 0
    processed_urls = 0

    for url in urls:
        logger.info(f"Processing URL: {url}")

        # Validate URL
        if not validate_url(url):
            logger.error(f"Invalid URL: {url}")
            continue

        # Scrape content
        scraped = web_scraper.scrape_url(url)
        if not scraped:
            logger.error(f"Failed to scrape URL: {url}")
            continue

        # Chunk the content
        content_chunks = chunk_text(scraped.content, max_chunk_size, overlap_size)

        # Create DocumentChunk objects
        document_chunks = []
        for i, chunk_content in enumerate(content_chunks):
            chunk_id = generate_chunk_id(scraped.url, i)
            doc_chunk = DocumentChunk(
                doc_path=scraped.url,
                heading=scraped.title,
                chunk_id=chunk_id,
                content=chunk_content
            )
            document_chunks.append(doc_chunk)

        # Add to vector store
        try:
            qdrant_client.add_document_chunks(document_chunks)
            logger.info(f"Added {len(document_chunks)} chunks from {url} to vector store")
            total_chunks += len(document_chunks)
            processed_urls += 1
        except Exception as e:
            logger.error(f"Error adding chunks from {url} to vector store: {str(e)}")

    logger.info(f"Completed processing. Total URLs processed: {processed_urls}, Total chunks created: {total_chunks}")

    return {
        "total_urls": len(urls),
        "processed_urls": processed_urls,
        "failed_urls": len(urls) - processed_urls,
        "total_chunks": total_chunks
    }


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Single-file URL ingestion pipeline")
    parser.add_argument("urls", nargs="+", help="URLs to process")
    parser.add_argument("--max_chunk_size", type=int, default=1000, help="Maximum size of each chunk")
    parser.add_argument("--overlap_size", type=int, default=200, help="Size of overlap between chunks")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be processed without actually ingesting")

    args = parser.parse_args()

    logger.info(f"Starting URL ingestion pipeline with {len(args.urls)} URLs")

    if args.dry_run:
        logger.info("DRY RUN MODE: Would process the following URLs:")
        for url in args.urls:
            print(f"  - {url}")
        logger.info(f"Chunk size: {args.max_chunk_size}, Overlap: {args.overlap_size}")
        return 0

    try:
        results = process_urls(
            urls=args.urls,
            max_chunk_size=args.max_chunk_size,
            overlap_size=args.overlap_size
        )

        print(f"Processing completed!")
        print(f"Total URLs: {results['total_urls']}")
        print(f"Successfully processed: {results['processed_urls']}")
        print(f"Failed: {results['failed_urls']}")
        print(f"Total chunks created: {results['total_chunks']}")

        return 0
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())