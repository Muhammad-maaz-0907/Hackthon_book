-- Database schema for Humanoid Robotics Book RAG Chatbot

-- Sessions table to track conversation sessions
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Interactions table to store chat interactions
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
);

-- Document index status table to track which documents are indexed
CREATE TABLE IF NOT EXISTS document_index_status (
    id SERIAL PRIMARY KEY,
    doc_path TEXT UNIQUE NOT NULL,
    last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT CHECK (status IN ('indexed', 'pending', 'failed')),
    chunk_count INTEGER DEFAULT 0
);

-- Feedback table to store user feedback on responses
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    interaction_id INTEGER NOT NULL,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id, interaction_id) REFERENCES interactions(session_id, id)
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_interactions_session_id ON interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at);