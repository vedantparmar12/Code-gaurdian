"""DuckDuckGo search integration - FREE, no API key required."""

import asyncio
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Check if duckduckgo-search is available
try:
    from duckduckgo_search import AsyncDDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False
    logger.warning("duckduckgo-search not installed. Install with: pip install duckduckgo-search")


class DuckDuckGoSearcher:
    """Free web search using DuckDuckGo - NO API KEY REQUIRED!"""

    def __init__(self):
        if not HAS_DDGS:
            raise ImportError(
                "duckduckgo-search not installed. "
                "Install with: pip install duckduckgo-search"
            )

    async def search_documentation_urls(
        self,
        library_name: str,
        version: Optional[str] = None,
        max_results: int = 10
    ) -> List[str]:
        """
        Search for documentation URLs using DuckDuckGo.

        Args:
            library_name: Name of the library
            version: Optional version
            max_results: Maximum results

        Returns:
            List of documentation URLs
        """
        try:
            queries = self._build_search_queries(library_name, version)
            all_urls = []

            async with AsyncDDGS() as ddgs:
                for query in queries:
                    logger.info(f"DuckDuckGo search: {query}")

                    # Search using DuckDuckGo
                    results = []
                    async for result in ddgs.text(
                        query,
                        max_results=5,
                        backend="api"  # Use API backend for better results
                    ):
                        results.append(result)

                    # Extract URLs
                    for result in results:
                        url = result.get('href') or result.get('link')
                        if url and self._is_documentation_url(url, library_name):
                            all_urls.append(url)

                    # Small delay to be respectful
                    await asyncio.sleep(0.5)

            # Remove duplicates
            unique_urls = list(dict.fromkeys(all_urls))

            # Prioritize official sources
            return self._prioritize_urls(unique_urls, library_name)[:max_results]

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    def _build_search_queries(self, library_name: str, version: Optional[str]) -> List[str]:
        """Build search queries."""
        queries = [
            f"{library_name} official documentation",
            f"{library_name} python docs",
            f"{library_name} readthedocs",
        ]

        if version:
            queries.insert(0, f"{library_name} {version} documentation")

        return queries[:3]  # Limit queries to avoid rate limiting

    def _is_documentation_url(self, url: str, library_name: str) -> bool:
        """Check if URL is likely a documentation page."""
        url_lower = url.lower()
        lib_lower = library_name.lower()

        # Documentation indicators
        doc_indicators = [
            'readthedocs.io',
            'docs.',
            '/docs/',
            '/documentation/',
            '/api/',
            '/reference/',
            'github.com',
            lib_lower,
        ]

        # Exclude non-documentation sites
        exclude_patterns = [
            'stackoverflow.com',
            'reddit.com',
            'twitter.com',
            'youtube.com',
            'pypi.org/project',  # PyPI package page, not docs
        ]

        # Check exclusions first
        if any(pattern in url_lower for pattern in exclude_patterns):
            return False

        # Check if it's a documentation URL
        return any(indicator in url_lower for indicator in doc_indicators)

    def _prioritize_urls(self, urls: List[str], library_name: str) -> List[str]:
        """Prioritize official documentation URLs."""
        def priority(url: str) -> int:
            url_lower = url.lower()
            lib_lower = library_name.lower()

            # Highest priority: Official docs
            if f'{lib_lower}.readthedocs.io' in url_lower:
                return 0
            if f'docs.{lib_lower}' in url_lower:
                return 1
            if f'{lib_lower}.org' in url_lower and 'docs' in url_lower:
                return 2

            # Medium priority: GitHub
            if 'github.com' in url_lower and lib_lower in url_lower:
                return 3

            # Lower priority: Other docs
            if 'readthedocs.io' in url_lower:
                return 4
            if '/docs/' in url_lower:
                return 5

            return 10

        return sorted(urls, key=priority)
