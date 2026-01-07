import asyncpg
import logging
from typing import List, Dict, Any, Optional
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class PostgresClient:
    def __init__(self):
        self.pool = None
        self.connection_string = os.getenv("NEON_DB_URL")

    async def init_db(self):
        """Initialize database connection and create required tables"""
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                dsn=self.connection_string,
                min_size=1,
                max_size=10,
                command_timeout=60
            )

            # Create required tables
            await self._create_tables()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    async def _create_tables(self):
        """Create necessary tables if they don't exist"""
        async with self.pool.acquire() as conn:
            # Create sessions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create interactions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    sources JSONB,
                    mode TEXT CHECK (mode IN ('rag', 'selected_text')),
                    confidence_score FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Create document_index_status table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_index_status (
                    id SERIAL PRIMARY KEY,
                    doc_path TEXT UNIQUE NOT NULL,
                    last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT CHECK (status IN ('indexed', 'pending', 'failed')),
                    chunk_count INTEGER DEFAULT 0
                )
            """)

            # Create feedback table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    interaction_id INTEGER NOT NULL,
                    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                    comment TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id, interaction_id) REFERENCES interactions(session_id, id)
                )
            """)

            logger.info("Tables created successfully")

    async def log_interaction(self, conversation_id: str, user_message: str,
                            bot_response: str, sources: List[Dict], mode: str):
        """Log a chat interaction to the database"""
        async with self.pool.acquire() as conn:
            # Ensure session exists
            await conn.execute("""
                INSERT INTO sessions (session_id)
                VALUES ($1)
                ON CONFLICT (session_id) DO NOTHING
            """, conversation_id)

            # Insert interaction
            await conn.execute("""
                INSERT INTO interactions (session_id, user_message, bot_response, sources, mode)
                VALUES ($1, $2, $3, $4, $5)
            """, conversation_id, user_message, bot_response, json.dumps(sources), mode)

            logger.info(f"Logged interaction for session {conversation_id}")

    async def get_conversation_history(self, conversation_id: str, limit: int = 10):
        """Retrieve conversation history"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT user_message, bot_response, timestamp
                FROM interactions
                WHERE session_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
            """, conversation_id, limit)

            return [
                {
                    "user_message": row["user_message"],
                    "bot_response": row["bot_response"],
                    "timestamp": row["timestamp"]
                }
                for row in rows
            ]

    async def update_document_index_status(self, doc_path: str, status: str, chunk_count: int = 0):
        """Update the indexing status of a document"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO document_index_status (doc_path, status, chunk_count)
                VALUES ($1, $2, $3)
                ON CONFLICT (doc_path)
                DO UPDATE SET
                    last_indexed = CURRENT_TIMESTAMP,
                    status = EXCLUDED.status,
                    chunk_count = EXCLUDED.chunk_count
            """, doc_path, status, chunk_count)

    async def get_indexed_documents(self):
        """Get list of indexed documents"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT doc_path, last_indexed, status, chunk_count
                FROM document_index_status
                WHERE status = 'indexed'
            """)

            return [
                {
                    "doc_path": row["doc_path"],
                    "last_indexed": row["last_indexed"],
                    "status": row["status"],
                    "chunk_count": row["chunk_count"]
                }
                for row in rows
            ]

    async def add_feedback(self, session_id: str, interaction_id: int, rating: int, comment: str = None):
        """Add feedback for an interaction"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO feedback (session_id, interaction_id, rating, comment)
                VALUES ($1, $2, $3, $4)
            """, session_id, interaction_id, rating, comment)

    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connections closed")