"""
Source Management with AI Summaries

Tracks and organizes documentation sources:
- AI-generated source summaries
- Source statistics and metadata
- Source filtering and organization
- Update tracking

Performance: Better organization and source-based filtering
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
import ollama

from config.settings import get_rag_config

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """Documentation source."""
    source_id: str
    url_pattern: str
    title: str
    summary: Optional[str] = None
    total_words: int = 0
    total_pages: int = 0
    last_crawled: Optional[datetime] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()


class SourceManager:
    """
    Manage documentation sources.

    Features:
    - Track sources with AI summaries
    - Update statistics
    - Filter by source
    - Organize documentation
    """

    def __init__(self, vector_store=None):
        """
        Initialize source manager.

        Args:
            vector_store: Supabase vector store instance
        """
        self.config = get_rag_config()
        self.vector_store = vector_store
        self.sources_cache: Dict[str, Source] = {}

        logger.info("Source manager initialized")

    async def add_source(
        self,
        source_id: str,
        url_pattern: str,
        title: str,
        generate_summary: bool = True
    ) -> Source:
        """
        Add a new documentation source.

        Args:
            source_id: Unique source identifier (e.g., "fastapi.tiangolo.com")
            url_pattern: URL pattern for this source
            title: Human-readable title
            generate_summary: Generate AI summary

        Returns:
            Source object
        """
        logger.info(f"Adding source: {source_id}")

        # Create source
        source = Source(
            source_id=source_id,
            url_pattern=url_pattern,
            title=title,
            created_at=datetime.now()
        )

        # Generate summary if enabled
        if generate_summary and self.config.use_agentic_rag:
            summary = await self._generate_source_summary(source_id, title, url_pattern)
            source.summary = summary

        # Store in database
        if self.vector_store:
            await self.vector_store.update_source(
                source_id=source_id,
                summary=source.summary or "",
                total_words=0
            )

        # Cache
        self.sources_cache[source_id] = source

        logger.info(f"Source added: {source_id}")
        return source

    async def update_source_stats(
        self,
        source_id: str,
        total_words: int,
        total_pages: int
    ):
        """
        Update source statistics.

        Args:
            source_id: Source ID
            total_words: Total words crawled
            total_pages: Total pages crawled
        """
        logger.debug(f"Updating stats for {source_id}: {total_pages} pages, {total_words} words")

        # Update cache
        if source_id in self.sources_cache:
            self.sources_cache[source_id].total_words = total_words
            self.sources_cache[source_id].total_pages = total_pages
            self.sources_cache[source_id].last_crawled = datetime.now()

        # Update database
        if self.vector_store:
            await self.vector_store.update_source(
                source_id=source_id,
                summary=self.sources_cache.get(source_id, Source(source_id, "", "")).summary or "",
                total_words=total_words
            )

    async def get_source(self, source_id: str) -> Optional[Source]:
        """
        Get source by ID.

        Args:
            source_id: Source ID

        Returns:
            Source object or None
        """
        # Check cache
        if source_id in self.sources_cache:
            return self.sources_cache[source_id]

        # Fetch from database
        if self.vector_store:
            try:
                result = await self.vector_store.supabase.table('sources').select('*').eq('source_id', source_id).single().execute()

                if result.data:
                    source = Source(
                        source_id=result.data['source_id'],
                        url_pattern=result.data.get('url_pattern', ''),
                        title=result.data.get('title', ''),
                        summary=result.data.get('summary'),
                        total_words=result.data.get('total_words', 0),
                        created_at=datetime.fromisoformat(result.data['created_at']) if result.data.get('created_at') else None
                    )

                    # Cache
                    self.sources_cache[source_id] = source
                    return source

            except Exception as e:
                logger.error(f"Failed to fetch source {source_id}: {e}")

        return None

    async def list_sources(self) -> List[Source]:
        """
        List all sources.

        Returns:
            List of Source objects
        """
        # If we have cached sources, return them
        if self.sources_cache:
            return list(self.sources_cache.values())

        # Fetch from database
        if self.vector_store:
            try:
                result = await self.vector_store.supabase.table('sources').select('*').execute()

                sources = []
                for row in result.data:
                    source = Source(
                        source_id=row['source_id'],
                        url_pattern=row.get('url_pattern', ''),
                        title=row.get('title', ''),
                        summary=row.get('summary'),
                        total_words=row.get('total_words', 0),
                        created_at=datetime.fromisoformat(row['created_at']) if row.get('created_at') else None
                    )
                    sources.append(source)
                    self.sources_cache[source.source_id] = source

                return sources

            except Exception as e:
                logger.error(f"Failed to list sources: {e}")

        return []

    async def _generate_source_summary(
        self,
        source_id: str,
        title: str,
        url_pattern: str
    ) -> str:
        """
        Generate AI summary for source.

        Args:
            source_id: Source ID
            title: Source title
            url_pattern: URL pattern

        Returns:
            AI-generated summary
        """
        logger.info(f"Generating summary for source: {source_id}")

        prompt = f"""Summarize this documentation source in 2-3 sentences:

Source: {title}
ID: {source_id}
URL Pattern: {url_pattern}

Describe:
1. What this documentation covers
2. Target audience (beginners/advanced developers)
3. Key topics or features documented

Keep it concise and informative."""

        try:
            # Generate summary using Ollama
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ollama.generate(
                    model=self.config.ollama_chat_model,
                    prompt=prompt
                )
            )

            summary = response.get('response', '').strip()

            if not summary:
                summary = f"Documentation for {title}"

            logger.info(f"Generated summary for {source_id}: {summary[:100]}...")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary for {source_id}: {e}")
            return f"Documentation for {title}"

    async def get_sources_by_pattern(self, pattern: str) -> List[Source]:
        """
        Find sources matching URL pattern.

        Args:
            pattern: URL pattern to match

        Returns:
            List of matching sources
        """
        all_sources = await self.list_sources()

        matching = [
            s for s in all_sources
            if pattern.lower() in s.url_pattern.lower() or pattern.lower() in s.title.lower()
        ]

        return matching

    async def suggest_sources(self, query: str, top_k: int = 3) -> List[Source]:
        """
        Suggest relevant sources for query.

        Args:
            query: User query
            top_k: Number of sources to suggest

        Returns:
            List of relevant sources
        """
        all_sources = await self.list_sources()

        if not all_sources:
            return []

        # Simple relevance scoring
        scored_sources = []

        for source in all_sources:
            score = 0.0

            # Match in title
            if query.lower() in source.title.lower():
                score += 1.0

            # Match in summary
            if source.summary and query.lower() in source.summary.lower():
                score += 0.5

            # Match in source_id
            if query.lower() in source.source_id.lower():
                score += 0.3

            if score > 0:
                scored_sources.append((score, source))

        # Sort by score
        scored_sources.sort(key=lambda x: x[0], reverse=True)

        # Return top_k
        return [s for _, s in scored_sources[:top_k]]

    def extract_source_id_from_url(self, url: str) -> str:
        """
        Extract source ID from URL.

        Args:
            url: Full URL

        Returns:
            Source ID (domain name)
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        domain = parsed.netloc

        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]

        return domain

    async def auto_add_source(
        self,
        url: str,
        title: str = None
    ) -> Source:
        """
        Automatically add source from URL.

        Args:
            url: URL to extract source from
            title: Optional title (auto-generated if not provided)

        Returns:
            Source object
        """
        source_id = self.extract_source_id_from_url(url)

        # Check if already exists
        existing = await self.get_source(source_id)
        if existing:
            logger.debug(f"Source {source_id} already exists")
            return existing

        # Auto-generate title if not provided
        if not title:
            title = source_id.replace('-', ' ').replace('.', ' ').title()

        # Extract URL pattern (base domain)
        from urllib.parse import urlparse
        parsed = urlparse(url)
        url_pattern = f"{parsed.scheme}://{parsed.netloc}"

        # Add source
        source = await self.add_source(
            source_id=source_id,
            url_pattern=url_pattern,
            title=title,
            generate_summary=True
        )

        return source


# ============ Utility Functions ============

def format_source_list(sources: List[Source]) -> str:
    """
    Format source list for display.

    Args:
        sources: List of sources

    Returns:
        Formatted string
    """
    if not sources:
        return "No sources found."

    lines = []
    for i, source in enumerate(sources, 1):
        lines.append(f"{i}. {source.title} ({source.source_id})")
        if source.summary:
            lines.append(f"   {source.summary}")
        if source.total_pages > 0:
            lines.append(f"   {source.total_pages} pages, {source.total_words:,} words")
        lines.append("")

    return "\n".join(lines)


def get_source_stats(sources: List[Source]) -> Dict[str, Any]:
    """
    Get aggregate source statistics.

    Args:
        sources: List of sources

    Returns:
        Statistics dict
    """
    total_pages = sum(s.total_pages for s in sources)
    total_words = sum(s.total_words for s in sources)

    return {
        'total_sources': len(sources),
        'total_pages': total_pages,
        'total_words': total_words,
        'avg_pages_per_source': total_pages / len(sources) if sources else 0,
        'avg_words_per_source': total_words / len(sources) if sources else 0
    }


# ============ Usage Example ============

async def demo_source_management():
    """Demonstrate source management."""

    print("Source Management Demo\n")

    # Create manager
    manager = SourceManager()

    # Add sources
    print("Adding sources...")

    fastapi = await manager.add_source(
        source_id="fastapi.tiangolo.com",
        url_pattern="https://fastapi.tiangolo.com",
        title="FastAPI Documentation",
        generate_summary=True
    )

    pydantic = await manager.add_source(
        source_id="docs.pydantic.dev",
        url_pattern="https://docs.pydantic.dev",
        title="Pydantic Documentation",
        generate_summary=True
    )

    print(f"\nAdded {len(manager.sources_cache)} sources")

    # Update stats
    print("\nUpdating statistics...")
    await manager.update_source_stats("fastapi.tiangolo.com", total_words=50000, total_pages=100)
    await manager.update_source_stats("docs.pydantic.dev", total_words=30000, total_pages=60)

    # List sources
    print("\nAll sources:")
    sources = await manager.list_sources()
    print(format_source_list(sources))

    # Suggest sources
    print("\nSuggesting sources for 'async validation':")
    suggested = await manager.suggest_sources("async validation", top_k=2)
    for s in suggested:
        print(f"- {s.title}: {s.summary}")

    # Stats
    stats = get_source_stats(sources)
    print(f"\nOverall stats:")
    print(f"  Total sources: {stats['total_sources']}")
    print(f"  Total pages: {stats['total_pages']}")
    print(f"  Total words: {stats['total_words']:,}")


if __name__ == "__main__":
    asyncio.run(demo_source_management())
