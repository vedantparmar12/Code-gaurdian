"""Universal documentation URL finder - works for ANY library without hardcoding."""

import asyncio
import logging
from typing import List, Optional
import httpx

logger = logging.getLogger(__name__)


class UniversalDocFinder:
    """Finds documentation for ANY Python library using multiple strategies."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def find_documentation_urls(
        self,
        library_name: str,
        max_urls: int = 10
    ) -> List[str]:
        """
        Dynamically find documentation URLs for ANY library.

        Strategies:
        1. PyPI API (gets actual project URLs)
        2. URL pattern probing (tries common patterns)
        3. GitHub search (if it's on GitHub)
        """
        library_name = library_name.lower().strip()
        all_urls = []

        # Strategy 1: PyPI API (most reliable!)
        pypi_urls = await self._get_urls_from_pypi(library_name)
        all_urls.extend(pypi_urls)

        # Strategy 2: Common URL patterns
        pattern_urls = await self._probe_common_patterns(library_name)
        all_urls.extend(pattern_urls)

        # Strategy 3: GitHub search
        github_urls = await self._search_github(library_name)
        all_urls.extend(github_urls)

        # Remove duplicates and validate
        unique_urls = []
        seen = set()
        for url in all_urls:
            if url not in seen and await self._validate_url(url):
                unique_urls.append(url)
                seen.add(url)

        logger.info(f"Found {len(unique_urls)} valid documentation URLs for {library_name}")
        return unique_urls[:max_urls]

    async def _get_urls_from_pypi(self, library_name: str) -> List[str]:
        """Get documentation URLs from PyPI API (official source!)."""
        urls = []
        try:
            # PyPI JSON API
            response = await self.client.get(f"https://pypi.org/pypi/{library_name}/json")

            if response.status_code == 200:
                data = response.json()
                info = data.get('info', {})

                # Get all URL fields from PyPI
                doc_urls = [
                    info.get('project_urls', {}).get('Documentation'),
                    info.get('project_urls', {}).get('Docs'),
                    info.get('project_urls', {}).get('documentation'),
                    info.get('home_page'),
                    info.get('docs_url'),
                    info.get('project_url'),
                ]

                # Filter out None and non-doc URLs
                for url in doc_urls:
                    if url and self._is_likely_documentation_url(url):
                        urls.append(url)

                # Also check project_urls for any doc-related entries
                project_urls = info.get('project_urls', {})
                for key, url in project_urls.items():
                    if any(keyword in key.lower() for keyword in ['doc', 'guide', 'tutorial', 'manual']):
                        if url and url not in urls:
                            urls.append(url)

                logger.info(f"PyPI API found {len(urls)} URLs for {library_name}")

        except Exception as e:
            logger.warning(f"PyPI API error for {library_name}: {e}")

        return urls

    async def _probe_common_patterns(self, library_name: str) -> List[str]:
        """Probe common documentation URL patterns."""
        patterns = [
            # ReadTheDocs
            f"https://{library_name}.readthedocs.io/en/latest/",
            f"https://{library_name}.readthedocs.io/en/stable/",
            f"https://{library_name}-python.readthedocs.io/en/latest/",

            # Custom domains
            f"https://{library_name}.org/docs/",
            f"https://{library_name}.org/documentation/",
            f"https://docs.{library_name}.org/",
            f"https://www.{library_name}.org/docs/",

            # GitHub Pages
            f"https://{library_name}.github.io/",
            f"https://{library_name}.github.io/docs/",

            # Subdomains
            f"https://docs.{library_name}.io/",
            f"https://{library_name}.docs.io/",
        ]

        valid_urls = []

        # Probe URLs in parallel
        tasks = [self._check_url_exists(url) for url in patterns]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, exists in zip(patterns, results):
            if exists is True:
                valid_urls.append(url)
                logger.info(f"Found valid URL: {url}")

        return valid_urls

    async def _search_github(self, library_name: str) -> List[str]:
        """Search for library on GitHub and find docs."""
        urls = []
        try:
            # GitHub search API (no auth needed for basic search)
            search_url = f"https://api.github.com/search/repositories"
            params = {
                "q": f"{library_name} language:python",
                "sort": "stars",
                "order": "desc",
                "per_page": 3
            }

            response = await self.client.get(search_url, params=params)

            if response.status_code == 200:
                data = response.json()
                for repo in data.get('items', [])[:3]:
                    # Check for documentation in repo
                    repo_name = repo.get('full_name')
                    default_branch = repo.get('default_branch', 'main')

                    # Possible doc locations
                    doc_urls = [
                        repo.get('homepage'),  # Project website
                        f"https://github.com/{repo_name}#readme",  # README
                        f"https://github.com/{repo_name}/blob/{default_branch}/README.md",
                        f"https://github.com/{repo_name}/tree/{default_branch}/docs",
                    ]

                    for url in doc_urls:
                        if url and url not in urls:
                            urls.append(url)

        except Exception as e:
            logger.warning(f"GitHub search error for {library_name}: {e}")

        return urls

    async def _check_url_exists(self, url: str) -> bool:
        """Check if a URL exists and is accessible."""
        try:
            response = await self.client.head(url, timeout=5.0)
            return response.status_code in [200, 301, 302, 307, 308]
        except:
            return False

    async def _validate_url(self, url: str) -> bool:
        """Validate that a URL is accessible and likely contains documentation."""
        try:
            response = await self.client.head(url, timeout=10.0)
            if response.status_code in [200, 301, 302, 307, 308]:
                # Additional check: URL should look like documentation
                return self._is_likely_documentation_url(url)
        except:
            pass
        return False

    def _is_likely_documentation_url(self, url: str) -> bool:
        """Check if URL is likely a documentation page."""
        if not url:
            return False

        url_lower = url.lower()

        # Documentation indicators
        doc_indicators = [
            'readthedocs.io',
            '/docs/',
            '/documentation/',
            '/doc/',
            '/api/',
            '/reference/',
            '/guide/',
            '/tutorial/',
            '/manual/',
            'github.io',
        ]

        # Exclude non-documentation
        exclude = [
            'pypi.org/pypi/',  # API endpoint, not docs
            '/issues/',
            '/pull/',
            '/releases/',
            '/blob/',  # Raw file, not docs page
        ]

        if any(ex in url_lower for ex in exclude):
            return False

        return any(ind in url_lower for ind in doc_indicators)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Convenience function
async def get_universal_doc_urls(library_name: str, max_urls: int = 10) -> List[str]:
    """
    Get documentation URLs for ANY library dynamically.

    This function:
    1. Checks PyPI API for official URLs
    2. Probes common documentation patterns
    3. Searches GitHub for the project
    4. Validates all URLs

    Returns only working, valid documentation URLs.
    """
    finder = UniversalDocFinder()
    try:
        return await finder.find_documentation_urls(library_name, max_urls)
    finally:
        await finder.close()
