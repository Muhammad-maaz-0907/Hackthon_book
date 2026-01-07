# Humanoid Robotics Book RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for the Humanoid Robotics Book, featuring both normal RAG mode and selected-text mode with citation support.

## Features

- **Normal RAG Mode**: Answers questions based on the entire book content with citations
- **Selected Text Mode**: Answers questions based only on user-selected text
- **Citation Support**: Shows sources (doc path + heading + chunk id) for all responses
- **Session Management**: Tracks conversation history
- **Logging**: Stores interactions and document indexing status
- **Feedback System**: Allows users to rate responses
- **Safety Checks**: Prevents hallucinations and prompt injection

## Architecture

- **Frontend**: Docusaurus-based documentation site with React chat component
- **Backend**: FastAPI server with async support
- **Vector Database**: Qdrant for document embeddings
- **Metadata Database**: Neon Postgres for logging and session management
- **LLM**: OpenAI GPT-4o for response generation
- **Embeddings**: Sentence Transformers for document chunk embeddings

## Installation

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- OpenAI API key
- Qdrant Cloud account (or self-hosted Qdrant)
- Neon Postgres account (or self-hosted Postgres)

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

### Frontend Setup

The chatbot is integrated into the Docusaurus site:

1. Install Node.js dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

## Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```bash
# Qdrant Configuration
QDRANT_URL=your-qdrant-url
QDRANT_API_KEY=your-qdrant-api-key

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key

# Neon Postgres Configuration
NEON_DB_URL=your-neon-db-url

# Application Configuration
DEBUG=false
LOG_LEVEL=INFO
```

### Safety Configuration

The system includes configurable safety levels in `safety_checker.py`:
- `STRICT`: Maximum safety, may be overly restrictive
- `MODERATE`: Balanced safety (default)
- `RELAXED`: Minimal safety checks

## Usage

### Starting the Backend

```bash
# Direct execution
uvicorn main:app --reload

# With Docker Compose
docker-compose up --build
```

### Indexing Documents

Before using the chatbot, index the book documents:

```bash
python index_documents.py
```

This will process all markdown files in the `docs/` directory and create embeddings in Qdrant.

### API Endpoints

- `POST /chat` - Main chat endpoint
  - Supports both RAG and selected-text modes
  - Request body: `{"message": "your question", "selected_text": "optional selected text"}`
  - Response includes: response text, sources, confidence score, conversation ID

- `GET /health` - Health check endpoint

### Frontend Integration

The chatbot component can be added to any Docusaurus page:

```jsx
import Chatbot from '@site/src/components/Chatbot';

function MyPage() {
  return (
    <div>
      <h1>My Content</h1>
      <Chatbot backendUrl="http://localhost:8000" />
    </div>
  );
}
```

## Modes of Operation

### Normal RAG Mode

- Searches the entire book for relevant information
- Returns citations with document path, heading, and chunk ID
- Ideal for general questions about the book content

### Selected Text Mode

- Answers questions based only on user-selected text
- Activate by selecting text on the page and using the "Select Text Mode" button
- Useful for focused questions about specific passages

## Safety Features

The system includes multiple layers of safety:

- **Prompt Injection Detection**: Identifies attempts to bypass instructions
- **Hallucination Prevention**: Validates responses against provided context
- **Content Safety**: Checks for potentially unsafe content in inputs
- **Confidence Scoring**: Provides confidence levels for responses

## Testing

Run the basic functionality test:

```bash
python test_main_functionality.py
```

For comprehensive testing, see the [TESTING.md](TESTING.md) guide.

## Deployment

For deployment options and production setup, see the [DEPLOYMENT.md](DEPLOYMENT.md) guide.

## Development

### Adding New Features

1. Update the backend API as needed
2. Add new UI components in `src/components/Chatbot/`
3. Update the frontend integration
4. Write tests for new functionality

### Document Processing

The system automatically processes markdown files in the `docs/` directory:
- Splits documents by headings
- Creates semantic chunks
- Generates embeddings
- Stores in vector database

## Troubleshooting

### Common Issues

1. **Connection Errors**: Verify environment variables and service availability
2. **Indexing Failures**: Check document formatting and permissions
3. **API Errors**: Confirm API keys and rate limits
4. **Performance Issues**: Monitor resource usage and scale as needed

### Debugging

Enable debug mode by setting `DEBUG=true` in your environment variables.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

[Specify your license here]

## Support

For issues and questions, please [create an issue](link-to-issues) in the repository.