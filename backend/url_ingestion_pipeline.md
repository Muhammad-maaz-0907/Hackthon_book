# Website URL Ingestion and Embedding Pipeline

This document describes the website URL ingestion and embedding pipeline that was implemented as part of the humanoid robotics book project.

## Overview

The pipeline allows users to ingest content from public website URLs and store it in the Qdrant vector database for use with the RAG chatbot. The system extracts clean textual content from websites, chunks it with overlap, generates semantic embeddings using Cohere models, and stores them in Qdrant.

## Components

### 1. Web Scraper (`web_scraper.py`)
- Extracts content from URLs using the `requests` library
- Uses BeautifulSoup to parse HTML and extract meaningful content
- Removes navigation, headers, footers, and other non-content elements
- Handles various HTML structures and content types

### 2. Cohere Client (`cohere_client.py`)
- Integrates with Cohere's embedding API
- Provides fallback to SentenceTransformer embeddings if Cohere fails
- Implements retry logic for handling rate limits
- Supports batch embedding for efficiency

### 3. URL Chunker (`url_chunker.py`)
- Chunks web content with configurable size and overlap
- Implements overlapping chunks to preserve context across splits
- Identifies headings and sections in web content
- Creates meaningful chunks based on content structure

### 4. URL Ingestor (`url_ingestor.py`)
- Coordinates the entire ingestion process
- Combines web scraping, chunking, and embedding
- Stores results in Qdrant vector database
- Provides status and error reporting

### 5. Enhanced Qdrant Client (`qdrant_client.py`)
- Updated to support both Cohere and SentenceTransformer embeddings
- Dynamic vector size based on embedding model used
- Maintains backward compatibility with existing functionality

## API Endpoint

### POST `/ingest-url`

Ingest content from a URL into the vector store.

**Request Body:**
```json
{
  "url": "https://example.com",
  "max_chunk_size": 1000,
  "overlap_size": 200
}
```

**Response:**
```json
{
  "success": true,
  "url": "https://example.com",
  "chunks_processed": 5,
  "chunks": ["chunk_id_1", "chunk_id_2", ...],
  "error": null
}
```

## Configuration

The pipeline uses the following environment variables:

- `COHERE_API_KEY`: Cohere API key for embeddings (required if using Cohere)
- `QDRANT_URL`: URL for Qdrant vector database
- `QDRANT_API_KEY`: API key for Qdrant database
- `USE_COHERE_EMBEDDINGS`: Whether to use Cohere embeddings (default: true)
- `WEB_SCRAPING_TIMEOUT`: Timeout for web requests in seconds (default: 30)
- `WEB_SCRAPING_USER_AGENT`: User agent string for web requests (default: "Humanoid-Robotics-Book-Bot/1.0")
- `DEFAULT_MAX_CHUNK_SIZE`: Default maximum chunk size (default: 1000)
- `DEFAULT_OVERLAP_SIZE`: Default overlap size (default: 200)

## Dependencies

The pipeline requires the following additional dependency:
- `cohere==5.5.3`

## Usage Example

```python
import requests

# Ingest a URL
response = requests.post("http://localhost:8000/ingest-url", json={
    "url": "https://docs.ros.org/en/humble/",
    "max_chunk_size": 1000,
    "overlap_size": 200
})

result = response.json()
print(f"Processed {result['chunks_processed']} chunks from {result['url']}")
```

## Features

- **Idempotent Operation**: Safe to run multiple times on the same URLs
- **Error Handling**: Comprehensive error handling and logging
- **Rate Limiting**: Respects target website rate limits
- **Fallback Embeddings**: Falls back to SentenceTransformer if Cohere is unavailable
- **Configurable Chunking**: Adjustable chunk size and overlap
- **Metadata Preservation**: Preserves URL, domain, and timestamp metadata