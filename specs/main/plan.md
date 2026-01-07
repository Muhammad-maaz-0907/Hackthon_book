# Implementation Plan: Single-File Website URL Ingestion and Embedding Pipeline

**Branch**: `main` | **Date**: 2026-01-04 | **Spec**: [F:\humanoid-robotics-book\specs\main\spec.md](file:///F:/humanoid-robotics-book/specs/main/spec.md)
**Input**: Feature specification from `/specs/main/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

The Single-File Website URL Ingestion and Embedding Pipeline will be implemented as a single executable Python file that ingests content from public website URLs, extracts clean textual content, chunks it deterministically with overlap, generates semantic embeddings using Cohere models, and stores them in Qdrant vector database. The system will support idempotent re-runs and comprehensive error logging, all within a single file for simplicity and ease of deployment.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: Cohere, BeautifulSoup4, Qdrant-client, SentenceTransformers, requests, python-dotenv
**Storage**: Qdrant vector database
**Testing**: pytest
**Target Platform**: Linux server
**Project Type**: single - single-file executable backend
**Performance Goals**: Process 10-50 URLs per minute depending on content size
**Constraints**: <200MB memory usage, single-file implementation, no web framework dependencies, offline-capable for existing content
**Scale/Scope**: Handle 1000+ URLs with millions of chunks

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Compliance with Physical AI & Humanoid Robotics Book Constitution:
- Content clarity: Target audience clearly defined as AI and backend engineers
- Consistency: Follows existing codebase patterns and architecture
- Actionable examples: Includes practical code samples and usage examples
- Progressive learning: Builds on existing RAG infrastructure concepts
- Accessibility: Code includes detailed comments and documentation
- Technical excellence: All code will be tested and validated

## Project Structure

### Documentation (this feature)

```text
specs/main/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── main.py              # Single-file implementation with all ingestion logic
├── pyproject.toml       # Project configuration and dependencies
└── .env.example         # Example environment variables file
```

**Structure Decision**: Single-file backend structure selected as per requirements. All ingestion logic consolidated in a single executable file with proper function organization and clear separation of concerns within the same file.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Single file with many functions | Required for simplicity and deployment | Multiple files would violate single-file constraint |
| Cohere + fallback strategy | Production reliability with fallback | Relying on single embedding provider could cause outages |
