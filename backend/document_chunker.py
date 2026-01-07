import os
import logging
from typing import List, Dict, Any
import re
from pathlib import Path
import markdown
from bs4 import BeautifulSoup
from .qdrant_client import QdrantClient, DocumentChunk

logger = logging.getLogger(__name__)

class DocumentChunker:
    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant_client = qdrant_client

    def chunk_documents(self, docs_dir: str) -> List[DocumentChunk]:
        """Chunk all markdown documents in the specified directory"""
        chunks = []
        docs_path = Path(docs_dir)

        # Walk through all markdown files in the docs directory
        for md_file in docs_path.rglob("*.md"):
            logger.info(f"Processing document: {md_file}")
            file_chunks = self._chunk_single_document(md_file)
            chunks.extend(file_chunks)

        logger.info(f"Total chunks created: {len(chunks)}")
        return chunks

    def _chunk_single_document(self, file_path: Path) -> List[DocumentChunk]:
        """Chunk a single markdown document"""
        chunks = []

        # Read the markdown file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Convert markdown to HTML to extract headings
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')

        # Extract document path relative to docs directory
        doc_path = str(file_path.relative_to(Path.cwd() / "docs"))

        # Split content by headings to create meaningful chunks
        sections = self._split_by_headings(content)

        for i, (heading, section_content) in enumerate(sections):
            # Create smaller chunks if the section is too large
            sub_chunks = self._create_sub_chunks(section_content, max_chunk_size=1000)

            for j, sub_chunk in enumerate(sub_chunks):
                chunk_id = f"{doc_path.replace('/', '_').replace('.', '_')}_{i}_{j}"

                chunk = DocumentChunk(
                    doc_path=doc_path,
                    heading=heading,
                    chunk_id=chunk_id,
                    content=sub_chunk
                )

                chunks.append(chunk)

        return chunks

    def _split_by_headings(self, content: str) -> List[tuple]:
        """Split content by markdown headings"""
        # Pattern to match markdown headings (h1 to h6)
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')

        sections = []
        current_section = []
        current_heading = "Introduction"  # Default heading if no heading is found at the start

        for line in lines:
            match = re.match(heading_pattern, line.strip())
            if match:
                # Save the previous section
                if current_section:
                    sections.append((current_heading, '\n'.join(current_section).strip()))

                # Start new section with this heading
                heading_level = len(match.group(1))
                heading_text = match.group(2)
                current_heading = heading_text
                current_section = [line]  # Include the heading line in the section
            else:
                current_section.append(line)

        # Don't forget the last section
        if current_section:
            sections.append((current_heading, '\n'.join(current_section).strip()))

        return sections

    def _create_sub_chunks(self, content: str, max_chunk_size: int = 1000) -> List[str]:
        """Create smaller chunks from larger content"""
        if len(content) <= max_chunk_size:
            return [content]

        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        sub_chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # If adding this paragraph would exceed the max size
            if len(current_chunk) + len(paragraph) > max_chunk_size:
                if current_chunk:
                    sub_chunks.append(current_chunk.strip())
                # Start a new chunk with this paragraph
                current_chunk = paragraph
            else:
                # Add this paragraph to the current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add the last chunk if it exists
        if current_chunk:
            sub_chunks.append(current_chunk.strip())

        # If any chunk is still too large, split by sentences
        final_chunks = []
        for chunk in sub_chunks:
            if len(chunk) <= max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Split large chunk by sentences
                sentence_chunks = self._split_by_sentences(chunk, max_chunk_size)
                final_chunks.extend(sentence_chunks)

        return final_chunks

    def _split_by_sentences(self, text: str, max_chunk_size: int) -> List[str]:
        """Split text into sentence-based chunks"""
        import re

        # Split by sentence endings
        sentences = re.split(r'[.!?]+\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Add period back to the sentence (except for the last one)
            if sentence != sentences[-1]:
                sentence += "."

            if len(current_chunk) + len(sentence) <= max_chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def index_documents(self, docs_dir: str):
        """Process and index all documents in the directory"""
        logger.info(f"Starting document indexing from: {docs_dir}")

        # Chunk documents
        chunks = self.chunk_documents(docs_dir)

        # Add chunks to Qdrant
        self.qdrant_client.add_document_chunks(chunks)

        logger.info(f"Successfully indexed {len(chunks)} chunks to Qdrant")

        return len(chunks)