# Quickstart: Single-File Website URL Ingestion and Embedding Pipeline

## Prerequisites

- Python 3.11+
- Qdrant vector database instance
- Cohere API key (optional, fallback to SentenceTransformer available)

## Setup

1. Create a `backend/` directory in your project root:
```bash
mkdir backend
cd backend
```

2. Install dependencies:
```bash
pip install cohere beautifulsoup4 qdrant-client sentence-transformers requests python-dotenv
```

3. Create a `.env` file with the following variables:

```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
COHERE_API_KEY=your_cohere_api_key  # Optional
USE_COHERE_EMBEDDINGS=true  # Set to false to use SentenceTransformer only
WEB_SCRAPING_TIMEOUT=30
DEFAULT_MAX_CHUNK_SIZE=1000
DEFAULT_OVERLAP_SIZE=200
```

## Usage

### Command Line Usage

```bash
# Ingest a single URL
python main.py ingest https://docs.ros.org/en/humble/

# Ingest multiple URLs
python main.py ingest https://example1.com https://example2.com

# With custom chunking parameters
python main.py ingest https://example.com --max_chunk_size 500 --overlap_size 100

# Dry run to see what would be processed
python main.py ingest https://example.com --dry_run
```

### Direct Python Usage

```python
from main import process_urls

# Process URLs directly
urls = ["https://docs.ros.org/en/humble/"]
results = process_urls(
    urls=urls,
    max_chunk_size=1000,
    overlap_size=200,
    use_cohere=True
)

print(f"Processed {results['total_chunks']} chunks from {results['total_urls']} URLs")
```

## Configuration

The pipeline supports the following environment variables:

- `COHERE_API_KEY`: Cohere API key for embeddings (optional, fallback available)
- `QDRANT_URL`: URL for Qdrant vector database
- `QDRANT_API_KEY`: API key for Qdrant database
- `USE_COHERE_EMBEDDINGS`: Whether to use Cohere embeddings (default: true)
- `WEB_SCRAPING_TIMEOUT`: Timeout for web requests in seconds (default: 30)
- `DEFAULT_MAX_CHUNK_SIZE`: Default maximum chunk size (default: 1000)
- `DEFAULT_OVERLAP_SIZE`: Default overlap size (default: 200)

## Project Structure

```
backend/
├── main.py              # Single-file implementation with all ingestion logic
├── .env                 # Environment variables (not committed)
├── .env.example         # Example environment variables
└── pyproject.toml       # Project configuration (optional)
```

## Verification

After ingestion, you can verify the content was stored by checking your Qdrant collection or implementing a retrieval function to test similarity search.