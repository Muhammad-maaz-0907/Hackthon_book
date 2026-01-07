# Tasks: Single-File Website URL Ingestion and Embedding Pipeline

**Feature**: Single-File Website URL Ingestion and Embedding Pipeline
**Date**: 2026-01-04
**Spec**: [F:\humanoid-robotics-book\specs\main\spec.md](file:///F:/humanoid-robotics-book/specs/main/spec.md)
**Plan**: [F:\humanoid-robotics-book\specs\main\plan.md](file:///F:/humanoid-robotics-book/specs/main/plan.md)

## Dependencies

- User Story 1 (URL ingestion and content extraction) must be completed before User Story 2 (Embedding generation)
- User Story 2 (Embedding generation) must be completed before User Story 3 (Vector storage)
- User Story 3 (Vector storage) must be completed before User Story 4 (Pipeline orchestration)

## Parallel Execution Examples

- Tasks T004 [P], T005 [P], T006 [P] can be developed in parallel as they implement different functions in main.py
- Tasks T008 [P], T009 [P] can be developed in parallel as they implement different aspects of the pipeline

## Implementation Strategy

**MVP Scope**: Tasks T001-T012 - Basic URL fetching, content extraction, and simple storage
**Incremental Delivery**:
1. Phase 1: Project setup and basic structure
2. Phase 2: Core ingestion functionality (fetch, extract, chunk)
3. Phase 3: Embedding generation
4. Phase 4: Vector storage
5. Phase 5: Full pipeline orchestration
6. Phase 6: Validation and polish

---

## Phase 1: Setup

### Goal
Initialize the project structure and dependencies for the single-file ingestion pipeline.

### Independent Test Criteria
- Project directory structure is created
- Dependencies can be installed
- Environment variables are properly configured

### Tasks

- [x] T001 Create backend directory at project root
- [x] T002 Create pyproject.toml with required dependencies for single-file implementation
- [x] T003 Create .env.example file with required environment variables
- [x] T004 Install dependencies: cohere, beautifulsoup4, qdrant-client, sentence-transformers, requests, python-dotenv

---

## Phase 2: Foundational

### Goal
Create the foundational configuration and structure for the single-file implementation.

### Independent Test Criteria
- Configuration can be loaded from environment variables
- Logging is properly configured
- Basic structure of main.py is established

### Tasks

- [x] T005 Define configuration constants and environment variable loading in single_file_ingestion.py
- [x] T006 Set up logging configuration in single_file_ingestion.py
- [x] T007 Create DocumentChunk data model in single_file_ingestion.py
- [x] T008 Create ScrapedContent data model in single_file_ingestion.py

---

## Phase 3: [US1] URL Ingestion and Content Extraction

### Goal
Implement the ability to fetch URLs and extract clean content from them.

### Independent Test Criteria
- URLs can be fetched successfully
- HTML content is properly parsed
- Clean text content is extracted
- Non-content elements (navigation, footer) are removed

### Tasks

- [x] T009 [US1] Implement URL validation function in backend/single_file_ingestion.py
- [x] T010 [US1] Implement URL fetching function in backend/single_file_ingestion.py
- [x] T011 [US1] Implement HTML parsing function in backend/single_file_ingestion.py
- [x] T012 [US1] Implement content extraction function in backend/single_file_ingestion.py
- [x] T013 [US1] Implement non-content element removal in backend/single_file_ingestion.py

---

## Phase 4: [US2] Text Normalization and Chunking

### Goal
Implement text normalization and deterministic chunking with overlap.

### Independent Test Criteria
- Text is normalized properly (whitespace, encoding)
- Content is split into chunks of configurable size
- Overlap is applied between chunks
- Stable chunk identifiers are generated

### Tasks

- [x] T014 [US2] Implement text normalization function in backend/single_file_ingestion.py
- [x] T015 [US2] Implement chunking function with configurable size in backend/single_file_ingestion.py
- [x] T016 [US2] Implement overlap logic between chunks in backend/single_file_ingestion.py
- [x] T017 [US2] Implement stable chunk identifier generation in backend/single_file_ingestion.py

---

## Phase 5: [US3] Metadata Enrichment

### Goal
Attach metadata to each chunk including source URL, title, and section information.

### Independent Test Criteria
- Source URL is attached to each chunk
- Page title and section headings are extracted and attached
- Metadata payload is properly formatted for vector storage

### Tasks

- [x] T018 [US3] Implement URL attachment to chunks in backend/single_file_ingestion.py
- [x] T019 [US3] Implement title extraction and attachment in backend/single_file_ingestion.py
- [x] T020 [US3] Implement section heading attachment in backend/single_file_ingestion.py
- [x] T021 [US3] Implement metadata payload preparation in backend/single_file_ingestion.py

---

## Phase 6: [US4] Embedding Generation

### Goal
Generate embeddings for text chunks using Cohere with fallback to SentenceTransformer.

### Independent Test Criteria
- Cohere client is initialized properly
- Embeddings are generated for text chunks
- Fallback to SentenceTransformer works when Cohere fails
- API errors and retries are handled properly

### Tasks

- [x] T022 [US4] Implement Cohere client initialization in backend/single_file_ingestion.py
- [x] T023 [US4] Implement embedding generation function using Cohere in backend/single_file_ingestion.py
- [x] T024 [US4] Implement fallback to SentenceTransformer in backend/single_file_ingestion.py
- [x] T025 [US4] Implement error handling and retries for embedding API in backend/single_file_ingestion.py

---

## Phase 7: [US5] Qdrant Integration

### Goal
Connect to Qdrant, create collection schema, and prepare upsert logic.

### Independent Test Criteria
- Qdrant client is initialized properly
- Collection schema is created or validated
- Vector size and payload structure are defined
- Upsert logic prevents duplicates

### Tasks

- [x] T026 [US5] Implement Qdrant client initialization in backend/single_file_ingestion.py
- [x] T027 [US5] Implement collection creation/validation in backend/single_file_ingestion.py
- [x] T028 [US5] Define vector size and payload structure in backend/single_file_ingestion.py
- [x] T029 [US5] Implement upsert logic to prevent duplicates in backend/single_file_ingestion.py

---

## Phase 8: [US6] Vector Storage

### Goal
Store embeddings with metadata in Qdrant vector database.

### Independent Test Criteria
- Embeddings are mapped to vector records
- Vectors with metadata are upserted into Qdrant
- Insertion results are logged properly
- No duplicate vectors are created

### Tasks

- [x] T030 [US6] Implement vector record mapping in backend/single_file_ingestion.py
- [x] T031 [US6] Implement upsert functionality to Qdrant in backend/single_file_ingestion.py
- [x] T032 [US6] Implement insertion result logging in backend/single_file_ingestion.py
- [x] T033 [US6] Verify no duplicates are created on re-run in backend/single_file_ingestion.py

---

## Phase 9: [US7] Pipeline Orchestration

### Goal
Create the main pipeline function that executes all stages in correct order.

### Independent Test Criteria
- Pipeline executes stages in correct order (fetch, extract, chunk, embed, store)
- Execution flow is deterministic
- Configuration parameters are properly passed between stages
- Pipeline can be executed as a command-line tool

### Tasks

- [x] T034 [US7] Implement main pipeline orchestration function in backend/single_file_ingestion.py
- [x] T035 [US7] Implement command-line argument parsing in backend/single_file_ingestion.py
- [x] T036 [US7] Ensure deterministic execution flow in backend/single_file_ingestion.py
- [x] T037 [US7] Implement pipeline execution as CLI tool in backend/single_file_ingestion.py

---

## Phase 10: [US8] Validation and Verification

### Goal
Implement validation checks to ensure pipeline works correctly and is idempotent.

### Independent Test Criteria
- Number of vectors stored is verified
- Basic similarity search sanity check passes
- Idempotent behavior confirmed on re-run
- Pipeline can be safely re-executed without creating duplicates

### Tasks

- [x] T038 [US8] Implement vector count verification in backend/single_file_ingestion.py
- [x] T039 [US8] Implement basic similarity search check in backend/single_file_ingestion.py
- [x] T040 [US8] Confirm idempotent behavior on re-run in backend/single_file_ingestion.py
- [x] T041 [US8] Test safe re-execution without duplicates in backend/single_file_ingestion.py

---

## Phase 11: [US9] Logging and Error Handling

### Goal
Implement comprehensive logging and error handling throughout the pipeline.

### Independent Test Criteria
- Stage-level log messages are added throughout pipeline
- Failures are surfaced clearly with actionable information
- Pipeline exits with non-zero status on fatal errors
- All error paths are handled gracefully

### Tasks

- [x] T042 [US9] Add stage-level logging throughout pipeline in backend/single_file_ingestion.py
- [x] T043 [US9] Implement clear error messaging in backend/single_file_ingestion.py
- [x] T044 [US9] Implement non-zero exit status on fatal errors in backend/single_file_ingestion.py
- [x] T045 [US9] Ensure all error paths are handled in backend/single_file_ingestion.py

---

## Phase 12: Polish & Cross-Cutting Concerns

### Goal
Document execution and finalize the implementation.

### Independent Test Criteria
- Usage instructions are documented
- Required environment variables are documented
- Re-running instructions are clear
- Implementation meets all success criteria from spec

### Tasks

- [x] T046 Document usage instructions in backend/single_file_ingestion.py comments
- [x] T047 Document required environment variables in README
- [x] T048 Describe how to re-run ingestion safely in README
- [x] T049 Verify all success criteria from spec are met