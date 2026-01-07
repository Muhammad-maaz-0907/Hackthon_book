import os
import logging
from typing import List, Dict, Any
import re
from pathlib import Path
from vector_client import QdrantClient, DocumentChunk
from web_scraper import WebScraper

logger = logging.getLogger(__name__)

class URLChunker:
    def __init__(self, qdrant_client: QdrantClient, web_scraper: WebScraper):
        self.qdrant_client = qdrant_client
        self.web_scraper = web_scraper

    def chunk_url_content(self, url: str, max_chunk_size: int = 1000, overlap_size: int = 200) -> List[DocumentChunk]:
        """
        Chunk content from a URL with overlap

        Args:
            url: URL to chunk
            max_chunk_size: Maximum size of each chunk
            overlap_size: Size of overlap between chunks

        Returns:
            List of DocumentChunk objects
        """
        # Scrape the URL content
        scraped_data = self.web_scraper.scrape_url(url)
        if not scraped_data:
            logger.error(f"Failed to scrape URL: {url}")
            return []

        content = scraped_data['content']
        title = scraped_data['title']
        domain = scraped_data['domain']

        chunks = []

        # Split content by headings/sections if possible
        sections = self._split_by_headings(content, title)

        for i, (heading, section_content) in enumerate(sections):
            # Create overlapping chunks from the section
            sub_chunks = self._create_overlapping_chunks(
                section_content,
                max_chunk_size=max_chunk_size,
                overlap_size=overlap_size
            )

            for j, sub_chunk in enumerate(sub_chunks):
                chunk_id = f"url_{domain.replace('.', '_')}_{i}_{j}_{int(scraped_data['timestamp'])}"

                chunk = DocumentChunk(
                    doc_path=f"url://{domain}{url.replace('https://', '').replace('http://', '')}",
                    heading=heading,
                    chunk_id=chunk_id,
                    content=sub_chunk,
                    page_number=None  # No page numbers for web content
                )

                chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks from URL: {url}")
        return chunks

    def _split_by_headings(self, content: str, default_title: str) -> List[tuple]:
        """Split content by potential headings/parts"""
        # For web content, we'll try to identify sections based on paragraph breaks and common patterns
        lines = content.split('\n')

        sections = []
        current_section = []
        current_heading = default_title or "Web Content"

        # Look for lines that might be headings (longer lines followed by shorter ones, or lines that seem like titles)
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # If line looks like a heading (shorter than surrounding lines and sentence-like)
            if self._is_heading_line(line, lines, i):
                if current_section:
                    sections.append((current_heading, '\n'.join(current_section).strip()))

                current_heading = line
                current_section = [line]
            else:
                current_section.append(line)

        # Don't forget the last section
        if current_section:
            sections.append((current_heading, '\n'.join(current_section).strip()))

        # If no sections were identified, create one with the default title
        if not sections:
            sections.append((default_title or "Web Content", content))

        return sections

    def _is_heading_line(self, line: str, all_lines: List[str], index: int) -> bool:
        """Determine if a line looks like a heading"""
        # A heading is typically shorter than surrounding content and might be followed by more detailed content
        if len(line) < 10 or len(line) > 100:  # Too short or too long to be a heading
            return False

        # Check if it looks like a sentence (starts with capital, ends with punctuation)
        if not line[0].isupper():
            return False

        # Check if the next few lines are longer (indicating content follows the heading)
        next_lines = []
        for i in range(index + 1, min(index + 4, len(all_lines))):
            next_line = all_lines[i].strip()
            if next_line:
                next_lines.append(next_line)

        if next_lines:
            avg_next_length = sum(len(l) for l in next_lines) / len(next_lines)
            if avg_next_length > len(line) * 1.5:  # Next lines are significantly longer
                return True

        # Check if line looks like a title (capitalized words, common heading patterns)
        words = line.split()
        if len(words) < 2 or len(words) > 10:  # Too few or too many words for a heading
            return False

        # Check for common heading patterns (all caps, title case, etc.)
        if line.isupper() and len(line) < 50:
            return True

        # If it has title-like capitalization
        title_words = [word for word in words if word[0].isupper()]
        if len(title_words) >= len(words) * 0.8:  # Most words are capitalized
            return True

        return False

    def _create_overlapping_chunks(self, content: str, max_chunk_size: int, overlap_size: int) -> List[str]:
        """Create overlapping chunks from content"""
        if len(content) <= max_chunk_size:
            return [content]

        # Split content into sentences
        sentences = self._split_into_sentences(content)

        chunks = []
        current_chunk = ""
        current_chunk_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # If adding this sentence would exceed the max size
            if current_chunk_size + sentence_size > max_chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # For overlapping chunks, start new chunk with overlap from previous chunk
                if overlap_size > 0 and chunks:
                    # Get the end portion of the previous chunk as overlap
                    prev_chunk = chunks[-1]
                    overlap_start_idx = max(0, len(prev_chunk) - overlap_size)
                    current_chunk = prev_chunk[overlap_start_idx:]
                    current_chunk_size = len(current_chunk)
                else:
                    current_chunk = ""
                    current_chunk_size = 0

            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_chunk_size += sentence_size + 1  # +1 for the space

        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Handle any remaining chunks that are still too large by splitting by paragraphs
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Split large chunk by paragraphs
                paragraph_chunks = self._split_by_paragraphs(chunk, max_chunk_size, overlap_size)
                final_chunks.extend(paragraph_chunks)

        return final_chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        # Split by sentence endings, preserving the punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _split_by_paragraphs(self, text: str, max_chunk_size: int, overlap_size: int) -> List[str]:
        """Split large text by paragraphs with overlap"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        current_chunk_size = 0

        for paragraph in paragraphs:
            paragraph_size = len(paragraph)

            if current_chunk_size + paragraph_size > max_chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Add overlap from previous chunk
                if overlap_size > 0 and chunks:
                    prev_chunk = chunks[-1]
                    overlap_start_idx = max(0, len(prev_chunk) - overlap_size)
                    current_chunk = prev_chunk[overlap_start_idx:]
                    current_chunk_size = len(current_chunk)
                else:
                    current_chunk = ""
                    current_chunk_size = 0

            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
            current_chunk_size += paragraph_size + 2  # +2 for the \n\n

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks