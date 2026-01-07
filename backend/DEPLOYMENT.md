# Deployment Guide for Humanoid Robotics Book RAG Chatbot

## Overview

This guide provides instructions for deploying the RAG chatbot backend for the Humanoid Robotics Book. The system consists of:
- FastAPI backend
- Qdrant vector database
- Neon Postgres database
- Docusaurus frontend integration

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Docusaurus    │───▶│   FastAPI       │───▶│   Qdrant         │
│   Frontend      │    │   Backend       │    │   Vector DB      │
│                 │    │                 │    │                  │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                            │
                            ▼
                    ┌──────────────────┐
                    │   Neon Postgres  │
                    │   Database       │
                    └──────────────────┘
```

## Prerequisites

- Docker and Docker Compose
- Access to OpenAI API
- Qdrant Cloud account (or self-hosted Qdrant)
- Neon Postgres account (or self-hosted Postgres)

## Environment Setup

Create a `.env` file in the backend directory:

```bash
# Qdrant Configuration
QDRANT_URL=https://your-cluster.qdrant.tech:6333
QDRANT_API_KEY=your-qdrant-api-key

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key

# Neon Postgres Configuration
NEON_DB_URL=postgresql://username:password@ep-xxx.us-east-1.aws.neon.tech/dbname?sslmode=require

# Application Configuration
DEBUG=false
LOG_LEVEL=INFO
```

## Local Development Deployment

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env with your actual credentials
```

### 3. Start Services with Docker Compose

```bash
docker-compose up --build
```

### 4. Index Documents

```bash
# In a separate terminal
cd backend
python index_documents.py
```

## Production Deployment

### Option 1: Docker Compose with External Services

If you're using external Qdrant and Postgres services:

```bash
# Create production environment file
cp .env.example .env.production
# Edit with production credentials

# Build and deploy
docker-compose -f docker-compose.prod.yml up -d
```

### Option 2: Kubernetes Deployment

Create Kubernetes manifests in `k8s/` directory:

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-chatbot-backend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: rag-chatbot-backend
  template:
    metadata:
      labels:
        app: rag-chatbot-backend
    spec:
      containers:
      - name: backend
        image: your-registry/rag-chatbot:latest
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: rag-chatbot-secrets
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### Option 3: Cloud Deployment (AWS/GCP/Azure)

#### AWS ECS Deployment

1. Build and push Docker image to ECR
2. Create ECS task definition
3. Deploy to ECS cluster

#### Heroku Deployment

1. Create `Procfile`:

```
web: uvicorn main:app --host=0.0.0.0 --port=${PORT:-8000}
```

2. Deploy:

```bash
heroku create your-app-name
heroku config:set QDRANT_URL=your-url
heroku config:set QDRANT_API_KEY=your-key
heroku config:set OPENAI_API_KEY=your-key
heroku config:set NEON_DB_URL=your-db-url
git push heroku main
```

## Document Indexing Process

### Initial Indexing

```bash
# Index all documents from the docs directory
cd backend
python index_documents.py
```

### Incremental Indexing

To re-index specific documents or add new ones:

```bash
# Re-index all documents
python index_documents.py

# The system automatically tracks indexing status in the document_index_status table
```

### Indexing Status Check

```sql
-- Check indexing status
SELECT doc_path, last_indexed, status, chunk_count
FROM document_index_status
ORDER BY last_indexed DESC;
```

## Docusaurus Integration

### Adding Chatbot to Pages

To add the chatbot to a Docusaurus page:

```jsx
import Chatbot from '@site/src/components/Chatbot';

function MyPage() {
  return (
    <div>
      <h1>My Content</h1>
      <Chatbot backendUrl="https://your-backend-url.com" />
    </div>
  );
}
```

### Adding to Layout

To add the chatbot to all pages, modify your layout component:

```jsx
// src/theme/Layout/index.js
import Chatbot from '@site/src/components/Chatbot';

export default function Layout(props) {
  return (
    <>
      <OriginalLayout {...props} />
      <div style={{ position: 'fixed', bottom: '20px', right: '20px', width: '400px' }}>
        <Chatbot backendUrl="https://your-backend-url.com" />
      </div>
    </>
  );
}
```

## Monitoring and Logging

### Health Checks

- `/health` - Basic health check
- Monitor response times and error rates
- Set up alerts for service degradation

### Logging

- Application logs are written to stdout
- Database queries are logged for debugging
- Performance metrics for response times

### Metrics

Monitor:
- API response times
- Database connection pool usage
- Vector search performance
- Error rates

## Security Considerations

### API Security

- Use HTTPS in production
- Implement rate limiting
- Validate all inputs
- Sanitize outputs to prevent XSS

### Data Security

- Encrypt sensitive data in transit and at rest
- Use environment variables for secrets
- Regular security audits
- Access control for admin endpoints

### Content Safety

- The system includes prompt injection detection
- Content filtering for unsafe inputs
- Regular review of safety rules

## Backup and Recovery

### Database Backup

```bash
# Back up Postgres database
pg_dump your-neon-db-url > backup.sql

# Restore from backup
psql your-neon-db-url < backup.sql
```

### Vector Database Backup

For Qdrant, use snapshots:

```bash
# Create snapshot
curl -X POST "http://your-qdrant-url/collections/humanoid_robotics_book/snapshots"

# Download snapshot
curl -X GET "http://your-qdrant-url/collections/humanoid_robotics_book/snapshots" -o snapshot.tar
```

## Troubleshooting

### Common Issues

1. **Connection Timeouts**: Check network connectivity to external services
2. **Memory Issues**: Increase container memory limits
3. **Indexing Failures**: Verify document format and permissions
4. **API Errors**: Check API keys and rate limits

### Debugging

Enable debug mode:
```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

### Performance Tuning

- Adjust the number of worker processes
- Optimize database queries
- Tune vector search parameters
- Implement caching for frequently accessed data

## Scaling

### Horizontal Scaling

- Use multiple backend instances behind a load balancer
- Scale database connection pool
- Use Redis for session management (optional)

### Vertical Scaling

- Increase instance size for CPU/memory intensive tasks
- Optimize embedding model usage
- Use faster vector databases

## Maintenance

### Regular Tasks

- Monitor indexing status
- Update dependencies regularly
- Review and clean up old sessions
- Check for security updates

### Updates

1. Pull latest code
2. Build new Docker image
3. Deploy with zero-downtime strategy
4. Verify functionality after deployment

## Support

For issues with deployment:
- Check logs for error messages
- Verify environment variables
- Ensure external services are accessible
- Contact support if issues persist