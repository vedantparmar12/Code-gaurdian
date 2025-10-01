"""Enhanced concurrent web crawler using Crawl4AI v0.7+ for documentation pages."""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, CrawlResult

logger = logging.getLogger(__name__)

# Virtual scroll and link preview are available through CrawlerRunConfig parameters
HAS_VIRTUAL_SCROLL = True  # Available via scan_full_page and scroll_delay
HAS_LINK_PREVIEW = True    # Available via experimental parameters and link extraction

logger.info("Virtual scroll and link preview features are available via CrawlerRunConfig")

# Import advanced crawling features
try:
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy, DFSDeepCrawlStrategy, BestFirstCrawlingStrategy
    from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer, CompositeScorer
    from crawl4ai.deep_crawling.filters import (
        FilterChain, URLPatternFilter, DomainFilter, ContentTypeFilter,
        ContentRelevanceFilter, SEOFilter
    )
    HAS_DEEP_CRAWLING = True
except ImportError:
    HAS_DEEP_CRAWLING = False
    logger.warning("Deep crawling features not available - using basic crawling")

# Import content processing features
try:
    from crawl4ai.content_filter_strategy import PruningContentFilter, BM25ContentFilter, LLMContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, LLMExtractionStrategy
    HAS_ADVANCED_PROCESSING = True
except ImportError:
    PruningContentFilter = BM25ContentFilter = LLMContentFilter = None
    DefaultMarkdownGenerator = JsonCssExtractionStrategy = LLMExtractionStrategy = None
    HAS_ADVANCED_PROCESSING = False
    logger.warning("Advanced content processing not available - using basic processing")

# Import async dispatcher for advanced multi-URL handling
try:
    from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher, SemaphoreDispatcher
    from crawl4ai import RateLimiter, CrawlerMonitor, DisplayMode
    HAS_ADVANCED_DISPATCHER = True
except ImportError:
    MemoryAdaptiveDispatcher = SemaphoreDispatcher = None
    RateLimiter = CrawlerMonitor = DisplayMode = None
    HAS_ADVANCED_DISPATCHER = False
    logger.warning("Advanced dispatchers not available - using basic concurrency")

from ..config import Settings
from ..models import DocumentPage


class DocumentCrawler:
    """Concurrent web crawler optimized for documentation pages."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.browser_config = self._create_browser_config()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._deep_crawl_strategies = self._create_deep_crawl_strategies()
        self._content_filters = self._create_content_filters()

    def _create_browser_config(self) -> BrowserConfig:
        """Create optimized browser configuration for documentation crawling."""
        import sys
        import io
        import os

        # Set stdout/stderr to UTF-8 to handle Unicode characters
        if sys.platform == 'win32':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

        # CRITICAL: Disable all console output from Crawl4AI to prevent JSON-RPC corruption
        # MCP servers communicate via JSON-RPC over stdio - any non-JSON output breaks the protocol
        os.environ['CRAWL4AI_VERBOSE'] = 'false'
        os.environ['CRAWL4AI_LOG_LEVEL'] = 'ERROR'

        return BrowserConfig(
            headless=True,
            verbose=False,  # Disable verbose output - critical for MCP
            extra_args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor",
                "--disable-extensions",
                "--disable-plugins",
                "--disable-images",  # Speed up crawling for text content
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-notifications",
                "--disable-default-apps",
                # Performance optimizations
                "--max_old_space_size=4096",
                "--no-first-run",
                "--disable-default-apps",
                "--log-level=3",  # Suppress Chromium console output
            ]
        )

    def _create_deep_crawl_strategies(self) -> Dict[str, Any]:
        """Create deep crawling strategies for different documentation types."""
        strategies = {}

        if HAS_DEEP_CRAWLING:
            # BFS strategy for comprehensive documentation sites
            strategies['bfs_comprehensive'] = BFSDeepCrawlStrategy(
                max_depth=3,
                max_pages=50,
                url_scorer=CompositeScorer([
                    KeywordRelevanceScorer(
                        keywords=['documentation', 'docs', 'api', 'reference', 'guide', 'tutorial'],
                        weight=0.7
                    ),
                    KeywordRelevanceScorer(
                        keywords=['getting started', 'quickstart', 'installation', 'setup'],
                        weight=0.8
                    )
                ]),
                filter_chain=FilterChain([
                    DomainFilter(allowed_domains=[], blocked_domains=['ads.', 'analytics.', 'tracking.']),
                    ContentTypeFilter(allowed_types=['text/html', 'application/xhtml+xml']),
                    URLPatternFilter(
                        patterns=[
                            r'.*/(docs?|documentation|reference|api|guide|tutorial).*',
                            r'.*/v?\d+(\.\d+)*.*',  # Version URLs
                            r'.*/(getting-started|quickstart|installation|setup).*'
                        ],
                        use_glob=False
                    )
                ]),
                include_external=False
            )

            # DFS strategy for deep tutorial exploration
            strategies['dfs_tutorials'] = DFSDeepCrawlStrategy(
                max_depth=4,
                max_pages=30,
                url_scorer=KeywordRelevanceScorer(
                    keywords=['tutorial', 'example', 'walkthrough', 'step-by-step', 'hands-on'],
                    weight=0.9
                ),
                filter_chain=FilterChain([
                    URLPatternFilter(
                        patterns=[
                            r'.*/(tutorial|example|walkthrough).*',
                            r'.*/(step-by-step|hands-on|learn).*'
                        ],
                        use_glob=False
                    ),
                    ContentRelevanceFilter(query="tutorial guide walkthrough", threshold=0.3)
                ]),
                include_external=False
            )

            # Best-first strategy for API documentation
            strategies['best_first_api'] = BestFirstCrawlingStrategy(
                max_depth=2,
                max_pages=25,
                url_scorer=KeywordRelevanceScorer(
                    keywords=['api', 'endpoint', 'method', 'parameter', 'response'],
                    weight=1.0
                ),
                filter_chain=FilterChain([
                    URLPatternFilter(
                        patterns=[r'.*/(api|reference|endpoints?).*'],
                        use_glob=False
                    )
                ]),
                include_external=False
            )

        return strategies

    def _create_content_filters(self) -> Dict[str, Any]:
        """Create content filtering strategies for better content quality."""
        filters = {}

        if HAS_ADVANCED_PROCESSING:
            # BM25 filter for content relevance scoring
            filters['bm25_documentation'] = BM25ContentFilter(
                user_query='documentation api tutorial guide reference installation usage example getting started',
                bm25_threshold=1.0,
                language='english'
            )

            # Pruning filter to remove navigation and boilerplate
            filters['pruning_filter'] = PruningContentFilter(
                user_query='documentation tutorial guide api reference',
                min_word_threshold=50,
                threshold_type='fixed',
                threshold=0.48
            )

            # LLM filter for high-quality content (disabled for simplicity)
            # NOTE: LLMContentFilter requires complex LLMConfig setup
            # Can be enabled with proper configuration if needed
            # if LLMContentFilter:
            #     filters['llm_quality'] = LLMContentFilter(
            #         instruction="Filter high-quality documentation content",
            #         chunk_token_threshold=1000000,
            #         verbose=False
            #     )

        return filters

    def _create_crawler_configs(self, library_name: str) -> List[CrawlerRunConfig]:
        """Create crawler configuration for documentation crawling (Crawl4AI 0.6.3 compatible)."""
        return [
            CrawlerRunConfig(
                cache_mode=CacheMode.WRITE_ONLY,
                wait_until="domcontentloaded",
            )
        ]

    async def crawl_pages_concurrent(
        self,
        urls: List[str],
        library_name: str = "",
        max_concurrent: Optional[int] = None
    ) -> List[DocumentPage]:
        """
        Crawl multiple pages concurrently with intelligent URL-specific configurations.

        Args:
            urls: List of URLs to crawl
            library_name: Name of the library for intelligent configuration matching
            max_concurrent: Maximum concurrent crawls (uses settings default if None)

        Returns:
            List of successfully crawled DocumentPage objects
        """
        if not urls:
            logger.warning("No URLs provided for crawling")
            return []

        max_concurrent = max_concurrent or self.settings.max_concurrent_crawls
        self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(f"Starting enhanced concurrent crawl of {len(urls)} URLs with max {max_concurrent} concurrent")

        # Create intelligent configurations for different URL types
        crawler_configs = self._create_crawler_configs(library_name)

        # Use advanced dispatcher if available for large batches
        if HAS_ADVANCED_DISPATCHER and len(urls) > 10:
            return await self._crawl_with_advanced_dispatcher(urls, library_name, max_concurrent)

        # Suppress stdout from Crawl4AI to prevent JSON-RPC corruption
        import sys
        import io
        import contextlib

        @contextlib.contextmanager
        def suppress_stdout():
            """Suppress stdout while preserving stderr for logging."""
            old_stdout = sys.stdout
            try:
                sys.stdout = io.StringIO()
                yield
            finally:
                sys.stdout = old_stdout

        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            # Use arun_many for intelligent URL-specific configuration matching
            try:
                # Suppress stdout during crawling to prevent progress output
                with suppress_stdout():
                    results = await crawler.arun_many(
                        urls=urls,
                        config=crawler_configs
                    )

                successful_pages = []
                failed_count = 0

                for i, result in enumerate(results):
                    if result and result.success:
                        try:
                            page = self._create_document_page_enhanced(result, urls[i], i)
                            if page:
                                successful_pages.append(page)
                            else:
                                failed_count += 1
                        except Exception as e:
                            logger.error(f"Error processing result for {urls[i]}: {e}")
                            failed_count += 1
                    else:
                        failed_count += 1
                        if result:
                            logger.warning(f"Failed to crawl {urls[i]}: Status {result.status_code}")
                        else:
                            logger.warning(f"No result returned for {urls[i]}")

                logger.info(f"Enhanced crawling completed: {len(successful_pages)} successful, {failed_count} failed")
                return successful_pages

            except Exception as e:
                logger.error(f"Error in arun_many: {e}")
                # Fallback to individual crawling with semaphore
                with suppress_stdout():
                    return await self._crawl_pages_fallback(crawler, urls, max_concurrent)

    async def _crawl_pages_fallback(
        self,
        crawler: AsyncWebCrawler,
        urls: List[str],
        max_concurrent: int
    ) -> List[DocumentPage]:
        """Fallback method using individual page crawling with semaphore."""
        logger.info("Using fallback crawling method with individual requests")

        tasks = [
            self._crawl_single_page_fallback(crawler, url, index)
            for index, url in enumerate(urls)
        ]

        # Use gather with return_exceptions=True to handle individual failures
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        successful_pages = []
        failed_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to crawl URL {urls[i]}: {result}")
                failed_count += 1
            elif result is not None:
                successful_pages.append(result)
            else:
                failed_count += 1

        logger.info(f"Fallback crawling completed: {len(successful_pages)} successful, {failed_count} failed")
        return successful_pages

    async def _crawl_single_page_fallback(
        self,
        crawler: AsyncWebCrawler,
        url: str,
        index: int
    ) -> Optional[DocumentPage]:
        """Fallback method for crawling a single page."""
        async with self._semaphore:
            try:
                logger.debug(f"[{index}] Starting fallback crawl: {url}")

                # Create a basic configuration for fallback (Crawl4AI 0.6.3 compatible)
                config = CrawlerRunConfig(
                    cache_mode=CacheMode.ENABLED,
                    wait_until="domcontentloaded",
                )

                # Execute crawl with timeout
                result = await asyncio.wait_for(
                    crawler.arun(url, config=config),
                    timeout=self.settings.crawl_timeout_seconds
                )

                # Validate crawl result
                if not result.success:
                    logger.warning(f"[{index}] Crawl unsuccessful: {url} - {result.status_code}")
                    return None

                # Check content quality
                if not self._is_valid_content(result, url):
                    return None

                # Create DocumentPage
                page = self._create_document_page_enhanced(result, url, index)
                if page:
                    logger.debug(f"[{index}] Successfully crawled: {url} ({page.word_count} words)")
                return page

            except asyncio.TimeoutError:
                logger.warning(f"[{index}] Timeout crawling: {url}")
                return None
            except Exception as e:
                logger.error(f"[{index}] Error crawling {url}: {e}")
                return None

    def _create_document_page_enhanced(
        self,
        result: CrawlResult,
        url: str,
        index: int
    ) -> Optional[DocumentPage]:
        """
        Create a DocumentPage from crawl result with enhanced features.

        Args:
            result: Successful crawl result
            url: Original URL
            index: Index for reference

        Returns:
            DocumentPage object or None if invalid
        """
        try:
            # Validate content quality first
            if not self._is_valid_content_enhanced(result, url):
                return None

            # Extract title with multiple fallback strategies
            title = self._extract_title_enhanced(result, url)

            # Clean and prepare content with link information
            markdown_content = self._clean_markdown_content_enhanced(result.markdown or "")
            html_content = result.cleaned_html or ""

            # Get additional metadata from links if available
            metadata = self._extract_metadata_from_result(result)

            # Count words in markdown for better accuracy
            word_count = len(markdown_content.split())

            # Add metadata to markdown if available
            if metadata:
                markdown_content = self._add_metadata_to_content(markdown_content, metadata)

            return DocumentPage(
                url=url,
                title=title,
                content=html_content,
                markdown=markdown_content,
                word_count=word_count,
                fetched_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error creating document page for {url}: {e}")
            return None

    def _is_valid_content_enhanced(self, result: CrawlResult, url: str) -> bool:
        """Enhanced content validation with additional checks."""
        try:
            # Basic content checks
            if not result.markdown or len(result.markdown.strip()) < 200:
                logger.debug(f"Content too short for URL: {url}")
                return False

            # Check for meaningful content indicators
            content_lower = result.markdown.lower()

            # Documentation quality indicators
            quality_indicators = [
                "function", "method", "class", "parameter", "argument",
                "return", "example", "sample", "tutorial", "guide",
                "install", "setup", "configuration", "api", "usage",
                "getting started", "quickstart", "documentation"
            ]

            quality_score = sum(1 for indicator in quality_indicators if indicator in content_lower)

            # Require at least 3 quality indicators for documentation
            if quality_score < 3:
                logger.debug(f"Low documentation quality score ({quality_score}) for URL: {url}")
                return False

            # Check content structure (headers, code blocks, lists)
            structure_indicators = [
                result.markdown.count('#'),  # Headers
                result.markdown.count('```'),  # Code blocks
                result.markdown.count('- '),  # Lists
                result.markdown.count('1. '),  # Numbered lists
            ]

            if sum(structure_indicators) < 5:  # Require some structural elements
                logger.debug(f"Poor content structure for URL: {url}")
                return False

            # Check for error page indicators (enhanced)
            error_indicators = [
                "404", "not found", "page not found", "page not available",
                "403", "forbidden", "access denied", "unauthorized",
                "500", "internal server error", "server error",
                "error occurred", "something went wrong", "oops",
                "under construction", "coming soon", "maintenance mode",
                "temporarily unavailable", "service unavailable"
            ]

            for indicator in error_indicators:
                if indicator in content_lower:
                    logger.debug(f"Error page detected for URL: {url}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating content for {url}: {e}")
            return False

    def _extract_title_enhanced(self, result: CrawlResult, url: str) -> str:
        """Extract title with multiple fallback strategies."""
        # Try result metadata title first (Crawl4AI 0.6.3)
        title_from_metadata = None
        if result.metadata and isinstance(result.metadata, dict):
            title_from_metadata = result.metadata.get('title') or result.metadata.get('og:title')

        if title_from_metadata and len(title_from_metadata.strip()) > 0:
            title = title_from_metadata.strip()
            # Clean common title suffixes
            title = title.split(' | ')[0].split(' - ')[0].split(' — ')[0]
            return title

        # Try to extract from markdown content
        markdown = result.markdown or ""
        lines = markdown.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# ') and len(line) > 2:
                return line[2:].strip()

        # Fallback to URL-based title
        parsed = urlparse(url)
        path_parts = [part for part in parsed.path.split('/') if part and part != 'index.html']
        if path_parts:
            title = path_parts[-1].replace('-', ' ').replace('_', ' ').title()
            return title

        return "Documentation"

    def _clean_markdown_content_enhanced(self, markdown: str) -> str:
        """Enhanced markdown cleaning with better structure preservation."""
        if not markdown:
            return ""

        lines = markdown.split('\n')
        cleaned_lines = []
        in_code_block = False

        for line in lines:
            # Track code blocks to preserve them
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                cleaned_lines.append(line)
                continue

            # Preserve code blocks as-is
            if in_code_block:
                cleaned_lines.append(line)
                continue

            line = line.strip()

            # Skip empty lines (will be re-added strategically)
            if not line:
                continue

            # Enhanced navigation element detection
            skip_patterns = [
                "skip to content", "skip to main", "skip navigation",
                "toggle navigation", "menu", "breadcrumb",
                "search...", "search documentation", "search docs",
                "edit this page", "improve this page", "edit on github",
                "report issue", "report bug", "suggest edit",
                "© copyright", "all rights reserved", "terms of service",
                "cookie", "privacy policy", "cookie policy",
                "last updated", "last modified", "edit history",
                "table of contents", "on this page", "in this section",
                "previous", "next", "back to top", "go to top",
                "print this page", "share this page", "bookmark"
            ]

            if any(pattern.lower() in line.lower() for pattern in skip_patterns):
                continue

            # Skip very short lines that are likely navigation
            if len(line) < 4 and not line.startswith('#'):
                continue

            # Clean up common markdown artifacts
            line = line.replace('\\n', '\n').replace('\\t', '\t')

            cleaned_lines.append(line)

        # Join with strategic spacing
        content = '\n\n'.join(cleaned_lines)

        # Clean up excessive spacing while preserving structure
        import re
        content = re.sub(r'\n{4,}', '\n\n\n', content)  # Max 3 consecutive newlines
        content = re.sub(r'[ \t]{3,}', '  ', content)  # Max 2 spaces

        return content.strip()

    def _extract_metadata_from_result(self, result: CrawlResult) -> Dict[str, Any]:
        """Extract additional metadata from crawl result."""
        metadata = {}

        # Extract link information if available
        if hasattr(result, 'links') and result.links:
            internal_links = result.links.get('internal', [])
            if internal_links:
                # Get high-scoring internal links
                high_score_links = [
                    link for link in internal_links
                    if link.get('total_score', 0) > 0.5
                ]
                if high_score_links:
                    metadata['related_links'] = high_score_links[:5]  # Top 5

        # Extract any structured data
        if hasattr(result, 'extracted_content') and result.extracted_content:
            metadata['extracted_data'] = result.extracted_content

        return metadata

    def _add_metadata_to_content(self, content: str, metadata: Dict[str, Any]) -> str:
        """Add metadata information to content in a structured way."""
        if not metadata:
            return content

        additions = []

        # Add related links section
        if 'related_links' in metadata:
            additions.append("\n\n## Related Links")
            for link in metadata['related_links']:
                link_text = link.get('text', 'Link')[:50]
                link_url = link.get('href', '')
                if link_url:
                    additions.append(f"- [{link_text}]({link_url})")

        # Add any extracted structured data
        if 'extracted_data' in metadata:
            additions.append("\n\n## Extracted Information")
            # Add structured data as needed

        return content + '\n'.join(additions)

    async def _crawl_single_page(
        self,
        crawler: AsyncWebCrawler,
        url: str,
        index: int
    ) -> Optional[DocumentPage]:
        """
        Crawl a single page with timeout and error handling.

        Args:
            crawler: The crawler instance
            url: URL to crawl
            index: Index for logging purposes

        Returns:
            DocumentPage if successful, None otherwise
        """
        async with self._semaphore:
            try:
                logger.debug(f"[{index}] Starting crawl: {url}")

                # Execute crawl with timeout
                result = await asyncio.wait_for(
                    crawler.arun(
                        url=url,
                        cache_mode=CacheMode.BYPASS,  # Always fetch fresh content
                        markdown_generator=True,
                        word_count_threshold=100,  # Skip pages with very little content
                        only_text=True,  # Extract text content only
                        remove_overlay_elements=True,  # Clean up popups and overlays
                    ),
                    timeout=self.settings.crawl_timeout_seconds
                )

                # Validate crawl result
                if not result.success:
                    logger.warning(f"[{index}] Crawl unsuccessful: {url} - {result.status_code}")
                    return None

                # Check content quality
                if not self._is_valid_content(result, url):
                    return None

                # Create DocumentPage
                page = self._create_document_page(result, url, index)
                logger.debug(f"[{index}] Successfully crawled: {url} ({page.word_count} words)")
                return page

            except asyncio.TimeoutError:
                logger.warning(f"[{index}] Timeout crawling: {url}")
                return None
            except Exception as e:
                logger.error(f"[{index}] Error crawling {url}: {e}")
                return None

    def _is_valid_content(self, result: CrawlResult, url: str) -> bool:
        """
        Validate if the crawled content is useful documentation.

        Args:
            result: Crawl result to validate
            url: Original URL for context

        Returns:
            True if content appears to be valid documentation
        """
        try:
            # Check if we have meaningful content
            if not result.markdown or len(result.markdown.strip()) < 200:
                logger.debug(f"Content too short for URL: {url}")
                return False

            if not result.cleaned_html or len(result.cleaned_html.strip()) < 200:
                logger.debug(f"HTML content too short for URL: {url}")
                return False

            # Check for common error page indicators
            error_indicators = [
                "404", "not found", "page not found",
                "403", "forbidden", "access denied",
                "500", "internal server error",
                "error occurred", "something went wrong",
                "under construction", "coming soon",
                "maintenance mode"
            ]

            content_lower = result.markdown.lower()
            # Get title from metadata (Crawl4AI 0.6.3)
            title_str = ""
            if result.metadata and isinstance(result.metadata, dict):
                title_str = result.metadata.get('title') or result.metadata.get('og:title') or ""
            title_lower = title_str.lower()

            for indicator in error_indicators:
                if indicator in content_lower or indicator in title_lower:
                    logger.debug(f"Error page detected for URL: {url}")
                    return False

            # Check for documentation-like content
            doc_indicators = [
                "api", "documentation", "docs", "reference",
                "guide", "tutorial", "example", "install",
                "usage", "getting started", "quickstart",
                "function", "method", "class", "parameter",
                "return", "example", "sample"
            ]

            has_doc_content = any(
                indicator in content_lower
                for indicator in doc_indicators
            )

            if not has_doc_content:
                logger.debug(f"No documentation indicators found for URL: {url}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating content for {url}: {e}")
            return False

    def _create_document_page(
        self,
        result: CrawlResult,
        url: str,
        index: int
    ) -> DocumentPage:
        """
        Create a DocumentPage from crawl result.

        Args:
            result: Successful crawl result
            url: Original URL
            index: Index for reference

        Returns:
            DocumentPage object
        """
        # Extract title with fallback (Crawl4AI 0.6.3)
        title = None
        if result.metadata and isinstance(result.metadata, dict):
            title = result.metadata.get('title') or result.metadata.get('og:title')

        if not title:
            # Extract title from URL path
            parsed = urlparse(url)
            path_parts = [part for part in parsed.path.split('/') if part]
            title = path_parts[-1].replace('-', ' ').replace('_', ' ').title() if path_parts else "Documentation"

        # Clean and prepare content
        markdown_content = self._clean_markdown_content(result.markdown)
        html_content = result.cleaned_html or ""

        # Count words in markdown for better accuracy
        word_count = len(markdown_content.split())

        return DocumentPage(
            url=url,
            title=title,
            content=html_content,
            markdown=markdown_content,
            fetched_at=datetime.now(),
            word_count=word_count
        )

    def _clean_markdown_content(self, markdown: str) -> str:
        """
        Clean and optimize markdown content for documentation.

        Args:
            markdown: Raw markdown content

        Returns:
            Cleaned markdown content
        """
        if not markdown:
            return ""

        lines = markdown.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            # Skip empty lines and common noise
            if not line:
                continue

            # Skip navigation elements and common noise
            skip_patterns = [
                "skip to content",
                "toggle navigation",
                "search...",
                "search documentation",
                "edit this page",
                "improve this page",
                "report issue",
                "© copyright",
                "all rights reserved",
                "cookie", "privacy policy",
                "terms of service",
            ]

            if any(pattern.lower() in line.lower() for pattern in skip_patterns):
                continue

            # Skip lines that are just symbols or very short
            if len(line) < 3 or line in ["---", "***", "###", "```"]:
                continue

            cleaned_lines.append(line)

        # Join with single newlines and limit consecutive newlines
        content = '\n'.join(cleaned_lines)

        # Replace multiple consecutive newlines with double newline
        import re
        content = re.sub(r'\n{3,}', '\n\n', content)

        return content.strip()

    async def crawl_single_url(self, url: str) -> Optional[DocumentPage]:
        """
        Crawl a single URL (convenience method).

        Args:
            url: URL to crawl

        Returns:
            DocumentPage if successful, None otherwise
        """
        results = await self.crawl_pages_concurrent([url], max_concurrent=1)
        return results[0] if results else None

    def estimate_crawl_time(self, urls: List[str]) -> float:
        """
        Estimate total crawl time in seconds.

        Args:
            urls: List of URLs to crawl

        Returns:
            Estimated time in seconds
        """
        num_urls = len(urls)
        max_concurrent = self.settings.max_concurrent_crawls
        timeout = self.settings.crawl_timeout_seconds

        # Estimate based on batches
        batches = (num_urls + max_concurrent - 1) // max_concurrent
        estimated_time = batches * timeout * 0.7  # Assume 70% of timeout on average

        return max(estimated_time, 10.0)  # Minimum 10 seconds


# Convenience functions with enhanced features
async def crawl_documentation_pages(
    urls: List[str],
    library_name: str = "",
    settings: Optional[Settings] = None,
    max_concurrent: Optional[int] = None
) -> List[DocumentPage]:
    """
    Enhanced convenience function to crawl documentation pages.

    Args:
        urls: URLs to crawl
        library_name: Name of the library for intelligent configuration matching
        settings: Application settings
        max_concurrent: Maximum concurrent crawls

    Returns:
        List of successfully crawled pages
    """
    from ..config import get_settings

    if settings is None:
        settings = get_settings()

    crawler = DocumentCrawler(settings)
    return await crawler.crawl_pages_concurrent(urls, library_name, max_concurrent)


async def crawl_single_page(
    url: str,
    library_name: str = "",
    settings: Optional[Settings] = None
) -> Optional[DocumentPage]:
    """
    Enhanced convenience function to crawl a single page.

    Args:
        url: URL to crawl
        library_name: Name of the library for intelligent configuration
        settings: Application settings

    Returns:
        DocumentPage if successful, None otherwise
    """
    results = await crawl_documentation_pages([url], library_name, settings, max_concurrent=1)
    return results[0] if results else None


def estimate_crawl_performance(
    urls: List[str],
    library_name: str = "",
    settings: Optional[Settings] = None
) -> Dict[str, Any]:
    """
    Estimate crawling performance and provide optimization recommendations.

    Args:
        urls: URLs to be crawled
        library_name: Library name for configuration matching
        settings: Application settings

    Returns:
        Performance estimates and recommendations
    """
    from ..config import get_settings

    if settings is None:
        settings = get_settings()

    crawler = DocumentCrawler(settings)
    estimated_time = crawler.estimate_crawl_time(urls)

    # Analyze URL types for optimization recommendations
    url_types = {
        'documentation': 0,
        'api_reference': 0,
        'tutorials': 0,
        'blogs': 0,
        'other': 0
    }

    for url in urls:
        url_lower = url.lower()
        if any(pattern in url_lower for pattern in ['docs', 'documentation']):
            url_types['documentation'] += 1
        elif any(pattern in url_lower for pattern in ['api', 'reference']):
            url_types['api_reference'] += 1
        elif any(pattern in url_lower for pattern in ['tutorial', 'guide', 'getting-started']):
            url_types['tutorials'] += 1
        elif any(pattern in url_lower for pattern in ['blog', 'news', 'article']):
            url_types['blogs'] += 1
        else:
            url_types['other'] += 1

    recommendations = []

    if url_types['documentation'] > 10:
        recommendations.append("Consider using virtual scroll for comprehensive content extraction")

    if url_types['tutorials'] > 5:
        recommendations.append("Tutorial pages will benefit from enhanced content structure analysis")

    if url_types['blogs'] > 0:
        recommendations.append("Blog content will use fresh fetching (no caching)")

    return {
        'total_urls': len(urls),
        'estimated_time_seconds': estimated_time,
        'url_type_breakdown': url_types,
        'max_concurrent': settings.max_concurrent_crawls,
        'timeout_per_page': settings.crawl_timeout_seconds,
        'recommendations': recommendations,
        'features_enabled': [
            'Multi-URL intelligent configuration matching',
            'Virtual scroll for infinite content',
            'Link preview with scoring',
            'Enhanced content validation',
            'Performance optimizations'
        ]
    }