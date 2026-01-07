import os
import logging
from typing import List, Optional
import cohere
from sentence_transformers import SentenceTransformer
import time

logger = logging.getLogger(__name__)

class CohereClient:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Cohere client with API key

        Args:
            api_key: Cohere API key (defaults to COHERE_API_KEY environment variable)
        """
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
        """
        Generate embedding for text using Cohere

        Args:
            text: Text to embed
            input_type: Type of input (search_query, search_document, etc.)

        Returns:
            List of embedding values
        """
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
        """
        Generate embeddings for multiple texts using Cohere

        Args:
            texts: List of texts to embed
            input_type: Type of input (search_query, search_document, etc.)

        Returns:
            List of embedding vectors
        """
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

    def embed_text_with_retry(self, text: str, input_type: str = "search_document", max_retries: int = 3) -> List[float]:
        """
        Generate embedding with retry logic for handling rate limits

        Args:
            text: Text to embed
            input_type: Type of input
            max_retries: Maximum number of retry attempts

        Returns:
            List of embedding values
        """
        for attempt in range(max_retries):
            try:
                response = self.client.embed(
                    texts=[text],
                    model=self.embedding_model,
                    input_type=input_type
                )
                return response.embeddings[0]
            except cohere.CohereAPIError as e:
                if "Too Many Requests" in str(e) and attempt < max_retries - 1:
                    # Rate limited, wait before retrying
                    wait_time = (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"Cohere embedding failed after {attempt + 1} attempts: {str(e)}, falling back to SentenceTransformer")
                    # Fallback to SentenceTransformer
                    embedding = self.fallback_model.encode(text)
                    return embedding.tolist()
            except Exception as e:
                logger.warning(f"Cohere embedding failed: {str(e)}, falling back to SentenceTransformer")
                # Fallback to SentenceTransformer
                embedding = self.fallback_model.encode(text)
                return embedding.tolist()

        # If all retries fail, use fallback
        embedding = self.fallback_model.encode(text)
        return embedding.tolist()

if __name__ == "__main__":
    # Example usage
    try:
        cohere_client = CohereClient()
        text = "This is a sample text for embedding."
        embedding = cohere_client.embed_text(text)
        print(f"Embedding length: {len(embedding)}")
        print(f"First 10 values: {embedding[:10]}")
    except ValueError as e:
        print(f"Error initializing Cohere client: {e}")
        print("Make sure to set COHERE_API_KEY environment variable")