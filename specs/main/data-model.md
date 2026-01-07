# Data Model: Single-File Website URL Ingestion and Embedding Pipeline

## Entities

### DocumentChunk
**Description**: Represents a chunk of content extracted from a URL
**Fields**:
- `doc_path`: string - URL or path identifier for the source
- `heading`: string - Heading/section where chunk was found
- `chunk_id`: string - Unique identifier for the chunk
- `content`: string - Content of the chunk
- `page_number`: Optional[int] - Page number if applicable (null for web content)

### ScrapedContent
**Description**: Internal model for scraped web content
**Fields**:
- `url`: string - Original URL
- `title`: string - Page title
- `domain`: string - Domain of the URL
- `content`: string - Extracted text content
- `timestamp`: int - Unix timestamp of scraping
- `status_code`: int - HTTP status code from scraping

## Relationships

- One URL ingestion request produces many DocumentChunk entities
- One ScrapedContent entity is processed into many DocumentChunk entities

## Validation Rules

- URL must be valid and accessible
- content must not be empty after processing
- chunk_id must be unique within the vector database
- max_chunk_size must be positive integer
- overlap_size must be non-negative and less than max_chunk_size

## State Transitions

- URL ingestion request: PENDING → PROCESSING → SUCCESS/FAILURE
- Document chunk: CREATED → EMBEDDED → STORED