# Spec 02: Single-File Website URL Ingestion and Embedding Pipeline

## Target Audience
AI and backend engineers implementing a simple, single-file ingestion pipeline for a RAG system.

## Focus
Create a single-file backend implementation that extracts textual content from published website URLs, generates semantic embeddings using Cohere models, and stores them in a vector database.

## Success Criteria
- Single executable Python file containing all ingestion logic
- Accepts a list of website URLs
- Extracts clean textual content
- Chunks text deterministically with overlap
- Preserves metadata (URL, title, section)
- Generates embeddings using Cohere
- Stores embeddings in Qdrant
- Supports idempotent re-runs
- Logs errors clearly

## Constraints
- Single-file implementation in `backend/main.py`
- Sources: Public Docusaurus URLs
- Embeddings: Cohere (latest stable)
- Vector DB: Qdrant Cloud Free Tier
- Secrets via environment variables
- No FastAPI or web framework dependencies
- Spec-Kit Plus structure

## Not Building
- Retrieval logic
- Agent orchestration
- FastAPI endpoints
- Frontend integration
- Multiple modules/packages

## Deliverables
- Single-file ingestion and embedding pipeline
- Vector DB collection setup
- Configuration via environment variables
- Minimal documentation