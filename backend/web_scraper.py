import os
import logging
import requests
from typing import Optional, Dict, Any
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time
import re

logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, timeout: int = 30, user_agent: Optional[str] = None):
        """
        Initialize the web scraper with configurable parameters

        Args:
            timeout: Request timeout in seconds
            user_agent: Custom user agent string
        """
        self.timeout = timeout
        self.user_agent = user_agent or "Humanoid-Robotics-Book-Bot/1.0"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

    def scrape_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape content from a given URL

        Args:
            url: URL to scrape

        Returns:
            Dictionary containing scraped content and metadata, or None if failed
        """
        try:
            # Validate URL format
            parsed_url = urlparse(url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                logger.error(f"Invalid URL format: {url}")
                return None

            # Make request to the URL
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract content
            title = self._extract_title(soup)
            text_content = self._extract_text_content(soup)
            domain = parsed_url.netloc

            # Clean up content
            cleaned_content = self._clean_content(text_content)

            return {
                'url': url,
                'title': title,
                'domain': domain,
                'content': cleaned_content,
                'timestamp': int(time.time()),
                'status_code': response.status_code
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error scraping URL {url}: {str(e)}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title from the HTML"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()

        # Try to find h1 as title if no title tag
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()

        return "Untitled"

    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from HTML, removing navigation and ads"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()

        # Look for main content containers (common patterns)
        main_content = None
        for selector in ['main', 'article', '[role="main"]', '.content', '#content', '.post', '.article']:
            main_content = soup.select_one(selector)
            if main_content:
                break

        # If no main content container found, use the body
        if not main_content:
            main_content = soup.find('body')

        # If still no content, use the full soup
        if not main_content:
            main_content = soup

        # Extract text, preserving paragraph structure
        paragraphs = []
        for element in main_content.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td']):
            text = element.get_text(separator=' ', strip=True)
            if text:
                paragraphs.append(text)

        return '\n\n'.join(paragraphs)

    def _clean_content(self, content: str) -> str:
        """Clean extracted content by removing extra whitespace and common artifacts"""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)

        # Remove common artifacts (e.g., multiple dots, special characters)
        content = re.sub(r'[.]{3,}', '...', content)
        content = content.strip()

        return content

if __name__ == "__main__":
    # Example usage
    scraper = WebScraper()
    result = scraper.scrape_url("https://docs.ros.org/en/humble/")
    if result:
        print(f"Title: {result['title']}")
        print(f"Domain: {result['domain']}")
        print(f"Content length: {len(result['content'])}")
        print(f"First 500 chars: {result['content'][:500]}...")
    else:
        print("Failed to scrape URL")