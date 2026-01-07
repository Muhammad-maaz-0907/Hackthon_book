from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the FastAPI app
app = FastAPI(
    title="Humanoid Robotics Book RAG Chatbot API",
    description="API for RAG-based chatbot with citation support for the Humanoid Robotics Book",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str = Field(..., description="User's question or message")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")
    selected_text: Optional[str] = Field(None, description="Optional selected text for selected-text mode")
    max_tokens: Optional[int] = Field(1000, description="Maximum tokens for response")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The chatbot's response")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="List of sources with citations")
    conversation_id: str = Field(..., description="Conversation ID for tracking")
    confidence: Optional[float] = Field(None, description="Confidence score of the response")

class DocumentChunk(BaseModel):
    doc_path: str = Field(..., description="Path to the document")
    heading: str = Field(..., description="Heading/section where chunk was found")
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., description="Content of the chunk")
    page_number: Optional[int] = Field(None, description="Page number if applicable")

# Import backend modules (will be implemented in other files)
from rag_engine import RAGEngine
from vector_client import QdrantClient
from postgres_client import PostgresClient
from selected_text_processor import SelectedTextProcessor
from url_ingestor import URLIngestor
from web_scraper import WebScraper
from config import load_config

# Initialize clients
qdrant_client = QdrantClient()
postgres_client = PostgresClient()
rag_engine = RAGEngine(qdrant_client, postgres_client)
selected_text_processor = SelectedTextProcessor()

# Initialize URL ingestor
url_ingestor = URLIngestor(qdrant_client)

@app.on_event("startup")
async def startup_event():
    """Initialize database connections and vector store on startup"""
    logger.info("Initializing RAG system...")
    await postgres_client.init_db()
    # Additional initialization can go here
    logger.info("RAG system initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down RAG system...")
    await postgres_client.close()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint that handles both normal RAG mode and selected-text mode
    """
    try:
        logger.info(f"Received chat request: {request.message[:50]}...")

        # Determine which mode to use
        if request.selected_text:
            # Selected text mode - only use provided text
            response = await selected_text_processor.process(
                user_message=request.message,
                selected_text=request.selected_text
            )
        else:
            # Normal RAG mode - retrieve from vector store
            response = await rag_engine.get_response(
                query=request.message,
                conversation_id=request.conversation_id,
                max_tokens=request.max_tokens
            )

        # Log the interaction
        await postgres_client.log_interaction(
            conversation_id=response.get('conversation_id'),
            user_message=request.message,
            bot_response=response.get('response', ''),
            sources=response.get('sources', []),
            mode='selected_text' if request.selected_text else 'rag'
        )

        return ChatResponse(**response)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

class URLIngestionRequest(BaseModel):
    url: str = Field(..., description="URL to ingest")
    max_chunk_size: Optional[int] = Field(1000, description="Maximum size of each chunk")
    overlap_size: Optional[int] = Field(200, description="Size of overlap between chunks")

class URLIngestionResponse(BaseModel):
    success: bool = Field(..., description="Whether the ingestion was successful")
    url: str = Field(..., description="URL that was ingested")
    chunks_processed: int = Field(..., description="Number of chunks created and stored")
    chunks: Optional[List[str]] = Field(None, description="List of chunk IDs created")
    error: Optional[str] = Field(None, description="Error message if ingestion failed")

@app.post("/ingest-url", response_model=URLIngestionResponse)
async def ingest_url_endpoint(request: URLIngestionRequest):
    """
    Ingest content from a URL into the vector store
    """
    try:
        logger.info(f"Received URL ingestion request for: {request.url}")

        # Ingest the URL
        result = url_ingestor.ingest_url(
            url=request.url,
            max_chunk_size=request.max_chunk_size,
            overlap_size=request.overlap_size
        )

        logger.info(f"URL ingestion completed for {request.url}: {result['chunks_processed']} chunks processed")

        return URLIngestionResponse(**result)

    except Exception as e:
        logger.error(f"Error in URL ingestion endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error during URL ingestion: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RAG Chatbot API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)