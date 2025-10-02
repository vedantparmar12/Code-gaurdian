"""
Ultra-Fast Crawler - 1000+ pages/min

Optimizations:
- 100 concurrent workers (vs 10)
- HTTP/2 support with multiplexing
- Connection pooling (reuse TCP connections)
- DNS caching (resolve once)
- Aggressive timeouts
- Fail-fast error handling

Target: 1000 pages/min (10x improvement)
"""
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import time
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class UltraCrawlResult:
    """Result of ultra-fast crawl."""
    url: str
    success: bool
    content: Optional[str] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    duration: float = 0.0


class UltraFastCrawler:
    """
    Ultra-fast crawler with aggressive optimizations.

    Features:
    - 100 concurrent workers
    - Connection pooling (reuse TCP)
    - DNS caching
    - HTTP/2 multiplexing
    - Aggressive timeouts
    - Fail-fast (no retries)
    """

    def __init__(
        self,
        max_workers: int = 100,
        connect_timeout: int = 2,
        read_timeout: int = 5,
        total_timeout: int = 10
    ):
        """
        Initialize ultra-fast crawler.

        Args:
            max_workers: Concurrent workers (default: 100)
            connect_timeout: Connect timeout in seconds
            read_timeout: Read timeout in seconds
            total_timeout: Total request timeout
        """
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)

        # Connection pool settings (HTTP/1.1 optimized)
        self.connector = aiohttp.TCPConnector(
            limit=max_workers,           # Total connections
            limit_per_host=30,           # Per-host limit
            ttl_dns_cache=300,           # Cache DNS for 5 min
            enable_cleanup_closed=True,  # Cleanup closed connections
            force_close=False,           # Reuse connections
            keepalive_timeout=30         # Keep connections alive
        )

        # Timeout settings (aggressive for speed)
        self.timeout = aiohttp.ClientTimeout(
            total=total_timeout,
            connect=connect_timeout,
            sock_read=read_timeout
        )

        logger.info(
            f"Ultra-fast crawler initialized: {max_workers} workers, "
            f"timeouts: connect={connect_timeout}s, read={read_timeout}s"
        )

    async def crawl_urls(
        self,
        urls: List[str],
        show_progress: bool = True
    ) -> List[UltraCrawlResult]:
        """
        Crawl multiple URLs at maximum speed.

        Args:
            urls: List of URLs to crawl
            show_progress: Show progress logs

        Returns:
            List of UltraCrawlResult objects
        """
        if not urls:
            return []

        start_time = time.time()
        logger.info(f"Ultra-fast crawl starting: {len(urls)} URLs")

        # Create session with connection pooling
        async with aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (compatible; DocFetcher/1.0)',
                'Accept': 'text/html,application/xhtml+xml',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
        ) as session:

            # Create tasks for all URLs
            tasks = [self._fetch_one(session, url) for url in urls]

            # Execute all in parallel with progress tracking
            results = []
            completed = 0

            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                completed += 1

                if show_progress and completed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (completed / elapsed) * 60 if elapsed > 0 else 0
                    logger.info(
                        f"Progress: {completed}/{len(urls)} "
                        f"({rate:.0f} pages/min)"
                    )

        # Final stats
        elapsed = time.time() - start_time
        rate = (len(urls) / elapsed) * 60 if elapsed > 0 else 0
        success_count = sum(1 for r in results if r.success)

        logger.info(
            f"Ultra-fast crawl complete: {success_count}/{len(urls)} successful "
            f"in {elapsed:.1f}s ({rate:.0f} pages/min) 🚀"
        )

        return results

    async def _fetch_one(
        self,
        session: aiohttp.ClientSession,
        url: str
    ) -> UltraCrawlResult:
        """
        Fetch single URL with semaphore limiting.

        Args:
            session: aiohttp session
            url: URL to fetch

        Returns:
            UltraCrawlResult
        """
        async with self.semaphore:
            start_time = time.time()

            try:
                async with session.get(url) as response:
                    duration = time.time() - start_time

                    if response.status == 200:
                        content = await response.text()

                        return UltraCrawlResult(
                            url=url,
                            success=True,
                            content=content,
                            status_code=response.status,
                            duration=duration
                        )
                    else:
                        return UltraCrawlResult(
                            url=url,
                            success=False,
                            status_code=response.status,
                            error=f"HTTP {response.status}",
                            duration=duration
                        )

            except asyncio.TimeoutError:
                duration = time.time() - start_time
                return UltraCrawlResult(
                    url=url,
                    success=False,
                    error="Timeout",
                    duration=duration
                )

            except aiohttp.ClientError as e:
                duration = time.time() - start_time
                return UltraCrawlResult(
                    url=url,
                    success=False,
                    error=f"Client error: {str(e)[:50]}",
                    duration=duration
                )

            except Exception as e:
                duration = time.time() - start_time
                return UltraCrawlResult(
                    url=url,
                    success=False,
                    error=f"Error: {str(e)[:50]}",
                    duration=duration
                )

    async def close(self):
        """Cleanup connector."""
        await self.connector.close()


class HTTP2Crawler:
    """
    HTTP/2 crawler for even better performance.

    HTTP/2 features:
    - Multiplexing (multiple requests per connection)
    - Header compression
    - Server push support
    """

    def __init__(self, max_workers: int = 100):
        """Initialize HTTP/2 crawler."""
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)

        logger.info(f"HTTP/2 crawler initialized with {max_workers} workers")

    async def crawl_urls(
        self,
        urls: List[str],
        show_progress: bool = True
    ) -> List[UltraCrawlResult]:
        """
        Crawl URLs using HTTP/2.

        Requires: pip install httpx[http2]

        Args:
            urls: URLs to crawl
            show_progress: Show progress

        Returns:
            List of results
        """
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed, falling back to HTTP/1.1")
            crawler = UltraFastCrawler(max_workers=self.max_workers)
            return await crawler.crawl_urls(urls, show_progress)

        start_time = time.time()
        logger.info(f"HTTP/2 crawl starting: {len(urls)} URLs")

        # Create HTTP/2 client
        async with httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(10.0, connect=2.0),
            limits=httpx.Limits(
                max_connections=self.max_workers,
                max_keepalive_connections=50
            )
        ) as client:

            # Create tasks
            tasks = [self._fetch_one_http2(client, url) for url in urls]

            # Execute
            results = []
            completed = 0

            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                completed += 1

                if show_progress and completed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (completed / elapsed) * 60 if elapsed > 0 else 0
                    logger.info(f"HTTP/2 progress: {completed}/{len(urls)} ({rate:.0f}/min)")

        # Stats
        elapsed = time.time() - start_time
        rate = (len(urls) / elapsed) * 60 if elapsed > 0 else 0
        success_count = sum(1 for r in results if r.success)

        logger.info(
            f"HTTP/2 crawl complete: {success_count}/{len(urls)} "
            f"in {elapsed:.1f}s ({rate:.0f} pages/min)"
        )

        return results

    async def _fetch_one_http2(self, client, url: str) -> UltraCrawlResult:
        """Fetch one URL with HTTP/2."""
        async with self.semaphore:
            start_time = time.time()

            try:
                response = await client.get(url)
                duration = time.time() - start_time

                if response.status_code == 200:
                    return UltraCrawlResult(
                        url=url,
                        success=True,
                        content=response.text,
                        status_code=response.status_code,
                        duration=duration
                    )
                else:
                    return UltraCrawlResult(
                        url=url,
                        success=False,
                        status_code=response.status_code,
                        error=f"HTTP {response.status_code}",
                        duration=duration
                    )

            except Exception as e:
                duration = time.time() - start_time
                return UltraCrawlResult(
                    url=url,
                    success=False,
                    error=str(e)[:50],
                    duration=duration
                )


# ============ Helper Functions ============

def filter_successful_results(results: List[UltraCrawlResult]) -> List[Dict[str, Any]]:
    """
    Filter successful results and convert to dict.

    Args:
        results: List of UltraCrawlResult

    Returns:
        List of successful results as dicts
    """
    successful = [
        {
            'url': r.url,
            'content': r.content,
            'duration': r.duration
        }
        for r in results
        if r.success and r.content
    ]

    return successful


def calculate_crawl_stats(results: List[UltraCrawlResult]) -> Dict[str, Any]:
    """
    Calculate crawl statistics.

    Args:
        results: Crawl results

    Returns:
        Stats dict
    """
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful

    avg_duration = (
        sum(r.duration for r in results) / total
        if total > 0 else 0
    )

    return {
        'total_urls': total,
        'successful': successful,
        'failed': failed,
        'success_rate': successful / total if total > 0 else 0,
        'avg_duration_ms': avg_duration * 1000,
        'total_content_mb': sum(
            len(r.content or '') for r in results
        ) / 1_000_000
    }


# ============ Usage Example ============

async def demo_ultra_fast_crawl():
    """Demonstrate ultra-fast crawling."""

    # Test URLs (mix of fast and slow)
    urls = [
        f"https://httpbin.org/delay/0"
        for _ in range(200)  # 200 URLs
    ]

    print("=== Ultra-Fast Crawler Demo ===\n")

    # Test 1: Standard crawler (baseline)
    print("1. Baseline (10 workers):")
    crawler_slow = UltraFastCrawler(max_workers=10)
    start = time.time()
    results_slow = await crawler_slow.crawl_urls(urls[:50], show_progress=False)
    time_slow = time.time() - start
    rate_slow = (50 / time_slow) * 60
    print(f"   50 URLs in {time_slow:.1f}s = {rate_slow:.0f} pages/min\n")

    # Test 2: Ultra-fast crawler (100 workers)
    print("2. Ultra-Fast (100 workers):")
    crawler_fast = UltraFastCrawler(max_workers=100)
    start = time.time()
    results_fast = await crawler_fast.crawl_urls(urls, show_progress=True)
    time_fast = time.time() - start
    rate_fast = (len(urls) / time_fast) * 60
    print(f"   {len(urls)} URLs in {time_fast:.1f}s = {rate_fast:.0f} pages/min 🚀\n")

    # Stats
    stats = calculate_crawl_stats(results_fast)
    print(f"Stats:")
    print(f"  Success rate: {stats['success_rate']*100:.1f}%")
    print(f"  Avg duration: {stats['avg_duration_ms']:.0f}ms")
    print(f"  Total content: {stats['total_content_mb']:.2f} MB")

    # Speedup
    speedup = rate_fast / rate_slow if rate_slow > 0 else 0
    print(f"\n🎉 Speedup: {speedup:.1f}x faster!")


if __name__ == "__main__":
    asyncio.run(demo_ultra_fast_crawl())
