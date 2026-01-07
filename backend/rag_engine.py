import logging
from typing import List, Dict, Any, Optional
import os
import uuid
from openai import AsyncOpenAI
import tiktoken
from qdrant_client import QdrantClient
from postgres_client import PostgresClient
from safety_checker import SafetyChecker, SafetyLevel

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, qdrant_client: QdrantClient, postgres_client: PostgresClient):
        self.qdrant_client = qdrant_client
        self.postgres_client = postgres_client

        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Initialize tokenizer for context management
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

        # Initialize safety checker
        self.safety_checker = SafetyChecker(safety_level=SafetyLevel.MODERATE)

    async def get_response(self, query: str, conversation_id: Optional[str] = None,
                          max_tokens: int = 1000, top_k: int = 5, min_score: float = 0.3) -> Dict[str, Any]:
        """Get response using RAG (Retrieval Augmented Generation)"""
        # Generate new conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        # Check query safety first
        query_safety = self.safety_checker.check_query_safety(query)
        if not query_safety["is_safe"]:
            logger.warning(f"Unsafe query detected: {query_safety['issues']}")
            return {
                "response": "I cannot process this query as it appears to contain unsafe instructions.",
                "sources": [],
                "conversation_id": conversation_id,
                "confidence": 0.0
            }

        # Search for relevant chunks
        search_results = self.qdrant_client.search(query, limit=top_k)

        # Filter results by minimum score
        relevant_chunks = [result for result in search_results if result['score'] >= min_score]

        if not relevant_chunks:
            return {
                "response": "I don't have enough information in the book to answer that question.",
                "sources": [],
                "conversation_id": conversation_id,
                "confidence": 0.0
            }

        # Prepare context from retrieved chunks
        context = self._prepare_context(relevant_chunks)

        # Check context relevance
        relevance_check = self.safety_checker.validate_context_relevance(query, context)
        if not relevance_check["is_relevant"]:
            logger.warning(f"Context relevance check failed: {relevance_check}")
            return {
                "response": "I don't have enough relevant information in the book to answer that question.",
                "sources": [],
                "conversation_id": conversation_id,
                "confidence": 0.0
            }

        # Build the prompt with context and query
        prompt = self._build_rag_prompt(context, query)

        # Generate response using OpenAI
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",  # Using gpt-4o for better performance
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant for the Humanoid Robotics Book. "
                            "Answer questions based only on the provided context from the book. "
                            "If the information is not in the provided context, say 'I don't have enough information in the book to answer that.' "
                            "Always cite sources when providing information from the context."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )

            # Extract the response text
            response_text = response.choices[0].message.content

            # Prepare sources with citations
            sources = self._format_sources(relevant_chunks)

            # Calculate confidence based on average score of retrieved chunks
            avg_score = sum(chunk['score'] for chunk in relevant_chunks) / len(relevant_chunks)

            # Apply safety filtering to the response
            filtered_result = self.safety_checker.filter_response(response_text, sources, avg_score)

            return {
                "response": filtered_result["response"],
                "sources": filtered_result["sources"],
                "conversation_id": conversation_id,
                "confidence": filtered_result["confidence"]
            }

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": "I encountered an error while processing your request. Please try again.",
                "sources": [],
                "conversation_id": conversation_id,
                "confidence": 0.0
            }

    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved chunks"""
        context_parts = []

        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"Source {i+1}:\n"
                f"Document: {chunk['doc_path']}\n"
                f"Section: {chunk['heading']}\n"
                f"Content: {chunk['content']}\n"
                f"Chunk ID: {chunk['chunk_id']}\n"
                f"Score: {chunk['score']}\n"
                "---\n"
            )

        return "\n".join(context_parts)

    def _build_rag_prompt(self, context: str, query: str) -> str:
        """Build the RAG prompt with context and query"""
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Instructions:\n"
            f"- Answer the question based only on the provided context\n"
            f"- If the context doesn't contain the information to answer the question, say 'I don't have enough information in the book to answer that.'\n"
            f"- Cite the sources you used to answer the question\n"
            f"- Be concise but comprehensive in your response\n"
            f"- If you reference specific content, mention the document path and section"
        )

        return prompt

    def _format_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format sources for response"""
        sources = []

        for chunk in chunks:
            source = {
                "doc_path": chunk["doc_path"],
                "heading": chunk["heading"],
                "chunk_id": chunk["chunk_id"],
                "score": chunk["score"],
                "content_preview": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"]
            }
            sources.append(source)

        return sources

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))