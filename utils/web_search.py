"""Web search integration with multiple backends."""

import asyncio
import logging
from typing import List, Optional
from urllib.parse import urlparse

import httpx
import ollama
from pydantic import BaseModel, HttpUrl, field_validator

from ..config import Settings

# Import alternative search methods
try:
    from .universal_doc_finder import get_universal_doc_urls
    HAS_UNIVERSAL_FINDER = True
except ImportError:
    HAS_UNIVERSAL_FINDER = False

try:
    from .duckduckgo_search import DuckDuckGoSearcher
    HAS_DUCKDUCKGO = True
except ImportError:
    HAS_DUCKDUCKGO = False

try:
    from .doc_url_registry import get_documentation_urls, list_registered_libraries
    HAS_REGISTRY = True
except ImportError:
    HAS_REGISTRY = False

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Single web search result."""

    title: str
    url: HttpUrl
    content: str

    @field_validator('content')
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Ensure content is not empty."""
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()


class WebSearcher:
    """Web search client using Ollama's API."""

    def __init__(self, settings: Settings):
        self.settings = settings
        # Set API key as environment variable BEFORE creating client
        import os
        os.environ['OLLAMA_API_KEY'] = settings.ollama_api_key
        # Configure ollama client with API key for web search
        self.ollama_client = ollama.Client()

    async def search_documentation_urls(
        self,
        library_name: str,
        version: Optional[str] = None,
        max_results: int = 10
    ) -> List[str]:
        """
        Search for official documentation URLs for a library.
        Uses multiple search strategies in order of preference:
        1. Universal Doc Finder (PyPI API + URL probing - works for ANY library!)
        2. DuckDuckGo (free, no rate limits)
        3. URL Registry (fallback for 40+ common libraries)
        4. Ollama Cloud API (last resort, has rate limits)

        Args:
            library_name: Name of the library to search for
            version: Optional version constraint
            max_results: Maximum number of URLs to return

        Returns:
            List of documentation URLs, prioritized by relevance
        """
        # Strategy 1: Universal Doc Finder (BEST - works for ANY library!)
        if HAS_UNIVERSAL_FINDER:
            try:
                logger.info(f"Using Universal Doc Finder for {library_name}...")
                universal_urls = await get_universal_doc_urls(library_name, max_results)
                if universal_urls:
                    logger.info(f"Universal Finder found {len(universal_urls)} URLs")
                    return universal_urls
            except Exception as e:
                logger.warning(f"Universal Doc Finder failed: {e}")

        # Strategy 2: Try DuckDuckGo (free, no rate limits)
        if HAS_DUCKDUCKGO:
            try:
                logger.info(f"Using DuckDuckGo search for {library_name}...")
                ddg = DuckDuckGoSearcher()
                ddg_urls = await ddg.search_documentation_urls(library_name, version, max_results)
                if ddg_urls:
                    logger.info(f"DuckDuckGo found {len(ddg_urls)} URLs")
                    return ddg_urls
            except Exception as e:
                logger.warning(f"DuckDuckGo search failed: {e}")

        # Strategy 3: Check URL registry (fallback for common libraries)
        if HAS_REGISTRY:
            logger.info(f"Checking URL registry for {library_name}...")
            registry_urls = get_documentation_urls(library_name, max_results)
            if registry_urls:
                logger.info(f"Found {len(registry_urls)} URLs in registry")
                return registry_urls

        # Strategy 4: Last resort - Ollama Cloud API (has rate limits)
        logger.info(f"Falling back to Ollama Cloud API for {library_name}...")
        return await self._search_with_ollama(library_name, version, max_results)

    async def _search_with_ollama(
        self,
        library_name: str,
        version: Optional[str] = None,
        max_results: int = 10
    ) -> List[str]:
        """Original Ollama Cloud search (kept as fallback)."""
        try:
            # Create targeted search queries for documentation
            queries = self._build_search_queries(library_name, version)

            all_urls = []
            for query in queries:
                logger.info(f"Searching with query: {query}")
                results = await self._execute_search(query, max_results=5)

                # Extract and validate URLs
                urls = self._extract_documentation_urls(results, library_name)
                all_urls.extend(urls)

                # Add delay between queries to respect rate limits
                await asyncio.sleep(1)

            # Remove duplicates while preserving order
            unique_urls = []
            seen = set()
            for url in all_urls:
                if url not in seen:
                    unique_urls.append(url)
                    seen.add(url)

            # Prioritize official sources
            prioritized_urls = self._prioritize_official_sources(unique_urls, library_name)

            return prioritized_urls[:max_results]

        except Exception as e:
            logger.error(f"Error searching for documentation URLs: {e}")
            return []

    def _build_search_queries(self, library_name: str, version: Optional[str]) -> List[str]:
        """Build targeted search queries for documentation."""
        base_queries = [
            f"{library_name} official documentation",
            f"{library_name} API reference docs",
            f"{library_name} developer guide documentation",
        ]

        if version:
            versioned_queries = [
                f"{library_name} {version} documentation",
                f"{library_name} v{version} API reference",
            ]
            return versioned_queries + base_queries

        # Add current year to get latest docs
        current_year_queries = [
            f"{library_name} documentation 2024 2025",
            f"{library_name} latest docs official site",
        ]

        return current_year_queries + base_queries

    async def _execute_search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Execute web search using Ollama API."""
        try:
            # Use ollama client's web_search method which handles authentication
            response = self.ollama_client.web_search(query, max_results=max_results)

            search_results = []
            # Response is a WebSearchResponse object with results attribute
            results = response.results if hasattr(response, 'results') else response.get('results', [])

            for result in results:
                try:
                    # Handle both dict and object responses
                    if hasattr(result, 'title'):
                        search_result = SearchResult(
                            title=result.title,
                            url=result.url,
                            content=result.content
                        )
                    else:
                        search_result = SearchResult(
                            title=result['title'],
                            url=result['url'],
                            content=result['content']
                        )
                    search_results.append(search_result)
                except Exception as e:
                    logger.warning(f"Invalid search result: {e}")
                    continue

            logger.info(f"Found {len(search_results)} results for query: {query}")
            return search_results

        except Exception as e:
            logger.error(f"Search execution failed for query '{query}': {e}")
            return []

    def _extract_documentation_urls(
        self,
        results: List[SearchResult],
        library_name: str
    ) -> List[str]:
        """Extract and validate documentation URLs from search results."""
        urls = []

        for result in results:
            url_str = str(result.url)

            # Basic validation
            if not self._is_valid_documentation_url(url_str, library_name):
                continue

            urls.append(url_str)

        return urls

    def _is_valid_documentation_url(self, url: str, library_name: str) -> bool:
        """Check if URL is likely to contain useful documentation."""
        try:
            parsed = urlparse(url)

            # Skip obviously non-documentation URLs
            skip_domains = {
                'github.com',  # We want docs sites, not GitHub repos directly
                'stackoverflow.com',
                'reddit.com',
                'twitter.com',
                'facebook.com',
                'linkedin.com',
                'youtube.com',
                'medium.com',  # Often outdated
            }

            if any(domain in parsed.netloc.lower() for domain in skip_domains):
                return False

            # Prefer URLs that contain the library name
            url_lower = url.lower()
            library_lower = library_name.lower()

            if library_lower in url_lower or library_lower.replace('-', '') in url_lower.replace('-', ''):
                return True

            # Check for documentation indicators
            doc_indicators = [
                'docs', 'documentation', 'api', 'reference',
                'guide', 'manual', 'tutorial', 'getting-started'
            ]

            return any(indicator in url_lower for indicator in doc_indicators)

        except Exception:
            return False

    def _prioritize_official_sources(self, urls: List[str], library_name: str) -> List[str]:
        """Prioritize official documentation sources."""
        official_patterns = [
            f"{library_name}.org",
            f"{library_name}.io",
            f"{library_name}.dev",
            f"{library_name}.com",
            f"docs.{library_name}",
            f"{library_name}.readthedocs.io",
            f"{library_name}.github.io",
        ]

        official_urls = []
        other_urls = []

        for url in urls:
            url_lower = url.lower()
            is_official = any(
                pattern.lower() in url_lower
                for pattern in official_patterns
            )

            if is_official:
                official_urls.append(url)
            else:
                other_urls.append(url)

        # Return official URLs first, then others
        return official_urls + other_urls

    async def fetch_page_content(self, url: str) -> Optional[str]:
        """
        Fetch page content using Ollama's web_fetch API.

        Args:
            url: URL to fetch

        Returns:
            Page content as markdown or None if failed
        """
        try:
            logger.info(f"Fetching page content from: {url}")

            # Use ollama.web_fetch for consistent fetching
            response = ollama.web_fetch(url)

            if response and response.get('content'):
                content = response['content']
                title = response.get('title', '')

                # Combine title and content
                if title:
                    return f"# {title}\n\n{content}"
                return content

            logger.warning(f"No content returned for URL: {url}")
            return None

        except Exception as e:
            logger.error(f"Error fetching page content from {url}: {e}")
            return None


async def search_library_documentation(
    library_name: str,
    version: Optional[str] = None,
    max_results: int = 10,
    settings: Optional[Settings] = None
) -> List[str]:
    """
    Convenience function to search for library documentation URLs.

    Args:
        library_name: Name of the library
        version: Optional version constraint
        max_results: Maximum URLs to return
        settings: Application settings

    Returns:
        List of documentation URLs
    """
    from ..config import get_settings

    if settings is None:
        settings = get_settings()

    searcher = WebSearcher(settings)
    return await searcher.search_documentation_urls(
        library_name=library_name,
        version=version,
        max_results=max_results
    )