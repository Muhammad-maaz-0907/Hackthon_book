---
id: 2
title: "Single-File Website URL Ingestion Pipeline"
stage: "plan"
date: "2026-01-04"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "main"
branch: "main"
user: "user"
command: "sp.plan"
labels: ["url-ingestion", "embedding", "pipeline", "web-scraping", "single-file"]
links:
  spec: "F:\\humanoid-robotics-book\\specs\\main\\spec.md"
  ticket: null
  adr: null
  pr: null
files:
  - "F:\\humanoid-robotics-book\\specs\\main\\spec.md"
  - "F:\\humanoid-robotics-book\\specs\\main\\plan.md"
  - "F:\\humanoid-robotics-book\\specs\\main\\research.md"
  - "F:\\humanoid-robotics-book\\specs\\main\\data-model.md"
  - "F:\\humanoid-robotics-book\\specs\\main\\quickstart.md"
  - "F:\\humanoid-robotics-book\\specs\\main\\contracts\\url-ingestion-cli.yaml"
tests: []
outcome: "Successfully created implementation plan for single-file website URL ingestion pipeline with all required artifacts"
evaluation: "Plan includes technical context, constitution check, project structure, and all supporting documents for single-file approach"
---

# Single-File Website URL Ingestion and Embedding Pipeline

Objective:
Translate Spec 01 into a concrete, staged implementation plan for a website URL ingestion and embedding pipeline, including initial backend project setup and a single-file execution model.

Planning goals:
- Define initial backend project structure and tooling
- Decompose the ingestion pipeline into clear, sequential stages
- Centralize all ingestion logic in a single executable entry file
- Ensure idempotency, observability, and clean stage boundaries
- Align all steps with Spec-Kit Plus conventions

Plan structure:

1. Initial project setup
   - Create a top-level `backend/` directory
   - Initialize the backend project using `uv` package manager
   - Configure Python version and dependency management via `pyproject.toml`
   - Set up environment variable handling for external services
   - Ensure the backend can be executed as a standalone module

2. Backend file structure
   - Inside `backend/`, create a single `main.py` file
   - All ingestion logic is implemented as functions within `main.py`
   - No additional modules or sub-packages are created at this stage

3. URL ingestion and content extraction
   - Implement a function to accept and validate website URLs
   - Fetch HTML content from each URL
   - Parse and extract primary textual content
   - Exclude navigation, footer, and non-content elements

4. Text cleaning and chunking
   - Implement a deterministic chunking function
   - Normalize and clean extracted text
   - Apply configurable chunk size and overlap
   - Associate chunks with stable identifiers

5. Embedding generation
   - Implement a function to generate embeddings using Cohere models
   - Batch chunks for efficient embedding requests
   - Handle API failures and retries gracefully

6. Vector storage integration
   - Implement a function to connect to Qdrant Cloud
   - Create or validate the target collection
   - Store embeddings with metadata using an upsert strategy
   - Prevent duplicate vectors on re-runs

7. Pipeline orchestration
   - Implement a main orchestration function that:
     - Fetches URLs
     - Extracts and chunks content
     - Generates embeddings
     - Stores vectors in Qdrant
   - Ensure execution order is explicit and deterministic

8. Validation and verification
   - Verify successful vector insertion counts
   - Perform basic similarity retrieval checks
   - Confirm idempotent behavior on repeated runs

9. Logging and error handling
   - Implement structured logging per pipeline stage
   - Surface actionable error messages for failures
   - Fail fast on unrecoverable errors

10. Documentation and readiness
    - Document execution steps and configuration
    - Describe extension points for retrieval and agent integration

Constraints:
- Single-file backend implementation (`backend/main.py`)
- No retrieval, agent, or API-layer logic
- No frontend or UI work
- No FastAPI integration in this spec

Output:
- A complete, staged implementation plan
- Explicit backend setup instructions
- Clear function-level responsibilities
- Ready-to-execute task breakdown for `/sp.tasks`

## Response

I have successfully created the implementation plan for the Single-File Website URL Ingestion and Embedding Pipeline. The following artifacts have been created:

1. **spec.md**: Feature specification for the single-file implementation
2. **plan.md**: Detailed implementation plan with technical context, constitution check, and project structure
3. **research.md**: Research findings with decisions and rationale for single-file architecture
4. **data-model.md**: Data model with entities, relationships, and validation rules
5. **quickstart.md**: Quickstart guide for using the single-file pipeline
6. **contracts/url-ingestion-cli.yaml**: Conceptual CLI interface specification
7. **PHR Record**: Complete prompt history record

The implementation plan focuses on a single-file approach where all ingestion logic is contained in `backend/main.py`, including:
- Web scraping functionality using requests and BeautifulSoup4
- Cohere embedding generation with SentenceTransformer fallback
- Text chunking with configurable size and overlap
- Qdrant vector storage integration
- Command-line interface for easy execution
- Comprehensive configuration via environment variables
- Proper error handling and logging

The plan maintains all requirements from the specification while adhering to the single-file constraint, ensuring simplicity and ease of deployment.