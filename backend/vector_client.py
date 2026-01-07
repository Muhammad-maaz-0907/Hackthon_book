import qdrant_client
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
import logging
from pydantic import BaseModel
import os
from sentence_transformers import SentenceTransformer
import uuid
from cohere_client import CohereClient

logger = logging.getLogger(__name__)

class DocumentChunk(BaseModel):
    doc_path: str
    heading: str
    chunk_id: str
    content: str
    page_number: Optional[int] = None

class QdrantClient:
    def __init__(self, use_cohere: bool = True):
        # Initialize Qdrant client
        self.client = qdrant_client.QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=True
        )

        # Initialize embedding models
        self.use_cohere = use_cohere
        self.sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

        if use_cohere:
            try:
                self.cohere_client = CohereClient()
                # Cohere embeddings are typically 1024-dimensional
                self.vector_size = 1024
            except ValueError as e:
                logger.warning(f"Could not initialize Cohere client: {e}. Falling back to SentenceTransformer.")
                self.use_cohere = False
                self.vector_size = 384  # Size for all-MiniLM-L6-v2
        else:
            self.vector_size = 384  # Size for all-MiniLM-L6-v2

        # Collection name for the book content
        self.collection_name = "humanoid_robotics_book"

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
                    size=self.vector_size,  # Dynamic size based on embedding model
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created collection '{self.collection_name}' with vector size {self.vector_size}")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using the configured model"""
        if self.use_cohere:
            embedding = self.cohere_client.embed_text(text)
        else:
            embedding = self.sentence_transformer_model.encode(text).tolist()
        return embedding

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

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks based on query"""
        query_vector = self.embed_text(query)

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

    def delete_collection(self):
        """Delete the entire collection (use with caution)"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")