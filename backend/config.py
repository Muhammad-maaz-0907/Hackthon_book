import os
from typing import Optional

class Config:
    """Configuration class for the URL ingestion pipeline"""

    # Qdrant configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")

    # Cohere configuration
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")

    # Web scraping configuration
    WEB_SCRAPING_TIMEOUT: int = int(os.getenv("WEB_SCRAPING_TIMEOUT", "30"))
    WEB_SCRAPING_USER_AGENT: str = os.getenv("WEB_SCRAPING_USER_AGENT", "Humanoid-Robotics-Book-Bot/1.0")

    # URL ingestion configuration
    URL_INGESTION_RATE_LIMIT: float = float(os.getenv("URL_INGESTION_RATE_LIMIT", "1.0"))  # requests per second
    DEFAULT_MAX_CHUNK_SIZE: int = int(os.getenv("DEFAULT_MAX_CHUNK_SIZE", "1000"))
    DEFAULT_OVERLAP_SIZE: int = int(os.getenv("DEFAULT_OVERLAP_SIZE", "200"))

    # Embedding configuration
    USE_COHERE_EMBEDDINGS: bool = os.getenv("USE_COHERE_EMBEDDINGS", "true").lower() == "true"

    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present"""
        required_vars = []

        if not cls.QDRANT_URL or not cls.QDRANT_API_KEY:
            required_vars.append("QDRANT_URL and QDRANT_API_KEY")

        if cls.USE_COHERE_EMBEDDINGS and not cls.COHERE_API_KEY:
            required_vars.append("COHERE_API_KEY (when USE_COHERE_EMBEDDINGS is true)")

        if required_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(required_vars)}")

        return True

def load_config() -> Config:
    """Load and validate configuration"""
    config = Config()
    config.validate()
    return config