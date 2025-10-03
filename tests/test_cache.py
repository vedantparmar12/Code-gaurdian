"""Tests for cache functionality."""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from typing import List
from unittest.mock import patch

import pytest
from pydantic import HttpUrl

from config import Settings
from mcp_doc_fetcher.models import LibraryDocumentation, DocumentPage, EmbeddingChunk
from mcp_doc_fetcher.utils.cache import DocumentCache


@pytest.fixture
def temp_settings():
    """Create settings with temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        temp_db_path = tmp.name

    settings = Settings(
        ollama_api_key="test_key",
        ollama_url="http://localhost:11434",
        ollama_embedding_model="nomic-embed-text",
        ollama_chat_model="gpt-oss:120b-cloud",
        cache_db_path=temp_db_path,
        cache_max_age_hours=24
    )

    yield settings

    # Cleanup
    if os.path.exists(temp_db_path):
        os.unlink(temp_db_path)


@pytest.fixture
async def cache(temp_settings):
    """Create and initialize cache instance."""
    cache_instance = DocumentCache(temp_settings)
    await cache_instance.initialize()
    yield cache_instance
    await cache_instance.close()


@pytest.fixture
def sample_documentation():
    """Create sample documentation for testing."""
    pages = [
        DocumentPage(
            url=HttpUrl("https://example.com/page1"),
            title="Getting Started",
            content="<h1>Getting Started</h1><p>This is the introduction...</p>",
            markdown="# Getting Started\n\nThis is the introduction...",
            word_count=50,
            fetched_at=datetime.now()
        ),
        DocumentPage(
            url=HttpUrl("https://example.com/page2"),
            title="API Reference",
            content="<h1>API Reference</h1><p>Function descriptions...</p>",
            markdown="# API Reference\n\nFunction descriptions...",
            word_count=75,
            fetched_at=datetime.now()
        )
    ]

    return LibraryDocumentation(
        library_name="testlib",
        version="1.0.0",
        pages=pages,
        total_pages=len(pages),
        cache_key="testlib:1.0.0",
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


@pytest.fixture
def sample_embeddings():
    """Create sample embedding chunks for testing."""
    return [
        EmbeddingChunk(
            page_id=0,
            chunk_index=0,
            text_content="This is the first chunk of text for testing.",
            token_count=10,
            embedding_vector=[0.1, 0.2, 0.3] + [0.0] * 765  # 768 dimensions
        ),
        EmbeddingChunk(
            page_id=0,
            chunk_index=1,
            text_content="This is the second chunk with different content.",
            token_count=12,
            embedding_vector=[0.4, 0.5, 0.6] + [0.0] * 765  # 768 dimensions
        ),
        EmbeddingChunk(
            page_id=1,
            chunk_index=0,
            text_content="API reference content with function descriptions.",
            token_count=8,
            embedding_vector=[0.7, 0.8, 0.9] + [0.0] * 765  # 768 dimensions
        )
    ]


class TestDocumentCache:
    """Test DocumentCache functionality."""

    @pytest.mark.asyncio
    async def test_initialize_cache(self, temp_settings):
        """Test cache initialization creates proper schema."""
        cache = DocumentCache(temp_settings)
        await cache.initialize()

        # Check that database file was created
        assert os.path.exists(temp_settings.cache_db_path)

        # Test can connect and query
        import aiosqlite
        async with aiosqlite.connect(temp_settings.cache_db_path) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in await cursor.fetchall()]

        expected_tables = {'libraries', 'pages', 'embeddings', 'cache_metadata'}
        assert expected_tables.issubset(set(tables))

        await cache.close()

    @pytest.mark.asyncio
    async def test_store_and_retrieve_documentation(
        self,
        cache,
        sample_documentation,
        sample_embeddings
    ):
        """Test storing and retrieving complete documentation."""
        # Store documentation
        success = await cache.store_library_documentation(
            sample_documentation,
            sample_embeddings
        )
        assert success

        # Retrieve documentation
        retrieved = await cache.get_library_documentation("testlib:1.0.0")
        assert retrieved is not None
        assert retrieved.library_name == "testlib"
        assert retrieved.version == "1.0.0"
        assert retrieved.total_pages == 2
        assert len(retrieved.pages) == 2

        # Check page details
        page1 = retrieved.pages[0]
        assert page1.title == "Getting Started"
        assert "introduction" in page1.markdown
        assert page1.word_count == 50

    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache, sample_documentation, sample_embeddings):
        """Test cache expiration functionality."""
        # Store documentation
        await cache.store_library_documentation(sample_documentation, sample_embeddings)

        # Modify settings to have very short cache expiration
        cache.settings.cache_max_age_hours = 0  # Expire immediately

        # Should return None due to expiration
        retrieved = await cache.get_library_documentation("testlib:1.0.0")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_search_embeddings(self, cache, sample_documentation, sample_embeddings):
        """Test semantic search functionality."""
        # Store documentation with embeddings
        await cache.store_library_documentation(sample_documentation, sample_embeddings)

        # Create a query embedding (similar to first chunk)
        query_embedding = [0.15, 0.25, 0.35] + [0.0] * 765

        # Search
        results = await cache.search_embeddings(
            library_name="testlib",
            query_embedding=query_embedding,
            max_results=3
        )

        assert len(results) > 0
        assert len(results) <= 3

        # Results should be tuples of (content, title, score)
        content, title, score = results[0]
        assert isinstance(content, str)
        assert isinstance(title, str)
        assert isinstance(score, float)
        assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_cache_stats(self, cache, sample_documentation, sample_embeddings):
        """Test cache statistics functionality."""
        # Initially empty cache
        stats = await cache.get_cache_stats()
        assert stats.total_libraries == 0
        assert stats.total_pages == 0

        # Store documentation
        await cache.store_library_documentation(sample_documentation, sample_embeddings)

        # Check updated stats
        stats = await cache.get_cache_stats()
        assert stats.total_libraries == 1
        assert stats.total_pages == 2
        assert stats.cache_size_mb > 0
        assert stats.newest_entry is not None

    @pytest.mark.asyncio
    async def test_clear_cache(self, cache, sample_documentation, sample_embeddings):
        """Test cache clearing functionality."""
        # Store documentation
        await cache.store_library_documentation(sample_documentation, sample_embeddings)

        # Verify it's stored
        retrieved = await cache.get_library_documentation("testlib:1.0.0")
        assert retrieved is not None

        # Clear specific library
        cleared = await cache.clear_cache("testlib")
        assert cleared >= 1

        # Verify it's gone
        retrieved = await cache.get_library_documentation("testlib:1.0.0")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_clear_all_cache(self, cache, sample_documentation, sample_embeddings):
        """Test clearing entire cache."""
        # Store documentation
        await cache.store_library_documentation(sample_documentation, sample_embeddings)

        # Clear all cache
        cleared = await cache.clear_cache()
        assert cleared >= 1

        # Verify cache is empty
        stats = await cache.get_cache_stats()
        assert stats.total_libraries == 0

    @pytest.mark.asyncio
    async def test_list_cached_libraries(self, cache, sample_documentation, sample_embeddings):
        """Test listing cached libraries."""
        # Initially empty
        libraries = await cache.list_cached_libraries()
        assert len(libraries) == 0

        # Store documentation
        await cache.store_library_documentation(sample_documentation, sample_embeddings)

        # List libraries
        libraries = await cache.list_cached_libraries()
        assert len(libraries) == 1

        lib = libraries[0]
        assert lib['name'] == 'testlib'
        assert lib['version'] == '1.0.0'
        assert lib['total_pages'] == 2
        assert 'created_at' in lib
        assert 'updated_at' in lib
        assert isinstance(lib['is_expired'], bool)

    @pytest.mark.asyncio
    async def test_upsert_library(self, cache, sample_documentation, sample_embeddings):
        """Test library record upsert functionality."""
        # Store initial documentation
        await cache.store_library_documentation(sample_documentation, sample_embeddings)

        # Update with new data
        updated_doc = sample_documentation.model_copy()
        updated_doc.total_pages = 3
        updated_doc.updated_at = datetime.now()

        await cache.store_library_documentation(updated_doc, sample_embeddings)

        # Should still be only one library, but updated
        libraries = await cache.list_cached_libraries()
        assert len(libraries) == 1
        assert libraries[0]['total_pages'] == 3

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, cache, sample_documentation, sample_embeddings):
        """Test concurrent cache operations."""
        # Create multiple documentation objects
        docs = []
        embeddings_list = []

        for i in range(3):
            doc = sample_documentation.model_copy()
            doc.library_name = f"testlib{i}"
            doc.cache_key = f"testlib{i}:1.0.0"
            docs.append(doc)

            # Create separate embeddings for each
            emb = [chunk.model_copy() for chunk in sample_embeddings]
            embeddings_list.append(emb)

        # Store concurrently
        tasks = [
            cache.store_library_documentation(doc, emb)
            for doc, emb in zip(docs, embeddings_list)
        ]

        results = await asyncio.gather(*tasks)
        assert all(results)  # All should succeed

        # Verify all were stored
        stats = await cache.get_cache_stats()
        assert stats.total_libraries == 3

    def test_serialize_deserialize_embedding(self, cache):
        """Test embedding vector serialization."""
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Serialize
        blob = cache._serialize_embedding_vector(vector)
        assert isinstance(blob, bytes)
        assert len(blob) > 0

        # Deserialize
        restored = cache._deserialize_embedding_vector(blob)
        assert len(restored) == len(vector)

        # Should be approximately equal (float precision)
        for orig, rest in zip(vector, restored):
            assert abs(orig - rest) < 1e-6


@pytest.mark.asyncio
async def test_convenience_function():
    """Test convenience function for getting cached documentation."""
    from mcp_doc_fetcher.utils.cache import get_cached_documentation

    with patch('mcp_doc_fetcher.config.get_settings') as mock_get_settings:
        mock_settings = Settings(
            ollama_api_key="test_key",
            ollama_url="http://localhost:11434",
            ollama_embedding_model="nomic-embed-text",
            ollama_chat_model="gpt-oss:120b-cloud",
            cache_db_path=":memory:"  # In-memory database
        )
        mock_get_settings.return_value = mock_settings

        # Should return None for non-existent documentation
        result = await get_cached_documentation("nonexistent", "1.0.0")
        assert result is None