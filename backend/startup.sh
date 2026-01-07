#!/bin/bash

# Wait for Qdrant to be ready
echo "Waiting for Qdrant to be ready..."
sleep 10

# Run document indexing
echo "Starting document indexing..."
python index_documents.py

# Start the FastAPI application
echo "Starting FastAPI application..."
exec uvicorn main:app --host 0.0.0.0 --port 8000