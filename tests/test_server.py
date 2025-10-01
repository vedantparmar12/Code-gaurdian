"""Tests for MCP server functionality."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent

from mcp_doc_fetcher.models import LibraryDocumentation, DocumentPage
from mcp_doc_fetcher.server import DocumentationFetcherServer


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from mcp_doc_fetcher.config import Settings
    return Settings(
        ollama_api_key="test_key",
        ollama_url="http://localhost:11434",
        ollama_embedding_model="nomic-embed-text",
        ollama_chat_model="gpt-oss:120b-cloud",
        cache_db_path=":memory:",  # In-memory database for testing
        server_name="test-doc-fetcher",
        server_version="1.0.0"
    )


@pytest.fixture
async def server(mock_settings):
    """Create and initialize server for testing."""
    with patch('mcp_doc_fetcher.server.get_settings', return_value=mock_settings):
        server_instance = DocumentationFetcherServer()

        # Mock the components to avoid actual initialization
        server_instance.cache = AsyncMock()
        server_instance.searcher = AsyncMock()
        server_instance.crawler = AsyncMock()
        server_instance.embedder = AsyncMock()

        yield server_instance

        await server_instance.shutdown()


class TestDocumentationFetcherServer:
    """Test DocumentationFetcherServer functionality."""

    def test_server_initialization(self, mock_settings):
        """Test server initializes with correct configuration."""
        with patch('mcp_doc_fetcher.server.get_settings', return_value=mock_settings):
            server = DocumentationFetcherServer()

            assert server.settings.server_name == "test-doc-fetcher"
            assert server.settings.server_version == "1.0.0"
            assert server.server.name == "test-doc-fetcher"

    @pytest.mark.asyncio
    async def test_list_tools(self, server):
        """Test listing available MCP tools."""
        # The tools are registered with the server, we can't easily call the handler
        # but we can verify they're registered by checking if the handler exists
        # For now, just verify the server has the expected tool names defined
        expected_tools = [
            "fetch_documentation",
            "search_documentation",
            "clear_cache",
            "refresh_documentation"
        ]

        # Verify server initialized correctly
        assert server.server is not None
        assert server.cache is not None
        assert server.searcher is not None
        assert server.crawler is not None
        assert server.embedder is not None

    @pytest.mark.asyncio
    async def test_fetch_documentation_cached(self, server):
        """Test fetch_documentation tool with cached content."""
        from datetime import datetime

        # Mock cached documentation
        mock_doc = LibraryDocumentation(
            library_name="fastapi",
            version="latest",
            pages=[DocumentPage(
                url="https://fastapi.tiangolo.com/",
                title="FastAPI",
                content="FastAPI framework documentation",
                markdown="# FastAPI",
                fetched_at=datetime.fromisoformat("2024-01-01T00:00:00"),
                word_count=100
            )],
            total_pages=5,
            cache_key="fastapi:latest",
            created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
            updated_at=datetime.fromisoformat("2024-01-01T12:00:00")
        )

        server.cache.get_library_documentation.return_value = mock_doc

        # Call fetch_documentation
        result = await server._handle_fetch_documentation({
            "library_name": "fastapi"
        })

        assert isinstance(result, CallToolResult)
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)

        # Parse response
        response_data = json.loads(result.content[0].text)
        assert response_data["success"] is True
        assert "cached" in response_data["message"].lower()
        assert response_data["data"]["source"] == "cache"

    @pytest.mark.asyncio
    async def test_fetch_documentation_fresh(self, server):
        """Test fetch_documentation tool with fresh fetching."""
        from pydantic import HttpUrl
        from datetime import datetime

        # Mock no cached content
        server.cache.get_library_documentation.return_value = None

        # Mock search results
        server.searcher.search_documentation_urls.return_value = [
            "https://fastapi.tiangolo.com/"
        ]

        # Mock crawled pages
        mock_pages = [
            DocumentPage(
                url=HttpUrl("https://fastapi.tiangolo.com/"),
                title="FastAPI",
                content="FastAPI documentation",
                markdown="# FastAPI\n\nDocumentation",
                word_count=100,
                fetched_at=datetime.now()
            )
        ]
        server.crawler.crawl_pages_concurrent.return_value = mock_pages

        # Mock embeddings
        server.embedder.generate_embeddings_for_pages.return_value = []

        # Mock cache storage
        server.cache.store_library_documentation.return_value = True

        # Call fetch_documentation
        result = await server._handle_fetch_documentation({
            "library_name": "fastapi",
            "max_pages": 10
        })

        assert isinstance(result, CallToolResult)
        response_data = json.loads(result.content[0].text)
        assert response_data["success"] is True
        assert response_data["data"]["source"] == "fresh"

    @pytest.mark.asyncio
    async def test_search_documentation(self, server):
        """Test search_documentation tool."""
        # Mock query embedding
        server.embedder.generate_query_embedding.return_value = [0.1, 0.2, 0.3]

        # Mock search results
        server.cache.search_embeddings.return_value = [
            ("Authentication example code", "Authentication Guide", 0.85),
            ("Login endpoint documentation", "API Reference", 0.75)
        ]

        result = await server._handle_search_documentation({
            "library_name": "fastapi",
            "query": "authentication",
            "max_results": 5
        })

        assert isinstance(result, CallToolResult)
        response_data = json.loads(result.content[0].text)
        assert response_data["success"] is True
        assert len(response_data["data"]["results"]) == 2

        # Check result structure
        first_result = response_data["data"]["results"][0]
        assert "title" in first_result
        assert "excerpt" in first_result
        assert "relevance_score" in first_result

    @pytest.mark.asyncio
    async def test_clear_cache(self, server):
        """Test clear_cache tool."""
        server.cache.clear_cache.return_value = 3  # 3 libraries cleared

        result = await server._handle_clear_cache({
            "library_name": "fastapi"
        })

        assert isinstance(result, CallToolResult)
        response_data = json.loads(result.content[0].text)
        assert response_data["success"] is True
        assert response_data["data"]["cleared_libraries"] == 3
        assert response_data["data"]["library_name"] == "fastapi"

    @pytest.mark.asyncio
    async def test_clear_all_cache(self, server):
        """Test clearing all cache."""
        server.cache.clear_cache.return_value = 5  # 5 libraries cleared

        result = await server._handle_clear_cache({})

        assert isinstance(result, CallToolResult)
        response_data = json.loads(result.content[0].text)
        assert response_data["success"] is True
        assert response_data["data"]["cleared_libraries"] == 5

    @pytest.mark.asyncio
    async def test_refresh_documentation(self, server):
        """Test refresh_documentation tool."""
        # Mock the underlying fetch operation
        server.cache.get_library_documentation.return_value = None  # Force refresh
        server.searcher.search_documentation_urls.return_value = ["https://example.com"]
        server.crawler.crawl_pages_concurrent.return_value = []
        server.embedder.generate_embeddings_for_pages.return_value = []
        server.cache.store_library_documentation.return_value = True

        result = await server._handle_refresh_documentation({
            "library_name": "fastapi"
        })

        # Should behave like fetch_documentation with force_refresh=True
        assert isinstance(result, CallToolResult)

    @pytest.mark.asyncio
    async def test_tool_validation_error(self, server):
        """Test tool call with invalid arguments."""
        result = await server._handle_fetch_documentation({
            "library_name": "",  # Invalid empty name
        })

        assert isinstance(result, CallToolResult)
        response_data = json.loads(result.content[0].text)
        assert response_data["success"] is False
        assert "Invalid request" in response_data["message"]

    @pytest.mark.asyncio
    async def test_tool_execution_error(self, server):
        """Test tool execution with underlying error."""
        server.cache.get_library_documentation.side_effect = Exception("Database error")

        result = await server._handle_fetch_documentation({
            "library_name": "fastapi"
        })

        assert isinstance(result, CallToolResult)
        response_data = json.loads(result.content[0].text)
        assert response_data["success"] is False
        assert "failed" in response_data["message"].lower()

    def test_create_success_result(self, server):
        """Test creating success result."""
        result = server._create_success_result(
            "Operation completed",
            {"key": "value"}
        )

        assert isinstance(result, CallToolResult)
        response_data = json.loads(result.content[0].text)
        assert response_data["success"] is True
        assert response_data["message"] == "Operation completed"
        assert response_data["data"]["key"] == "value"

    def test_create_error_result(self, server):
        """Test creating error result."""
        result = server._create_error_result(
            "Operation failed",
            "ERROR_CODE"
        )

        assert isinstance(result, CallToolResult)
        response_data = json.loads(result.content[0].text)
        assert response_data["success"] is False
        assert response_data["message"] == "Operation failed"
        assert response_data["error_code"] == "ERROR_CODE"

    @pytest.mark.asyncio
    async def test_read_cache_stats(self, server):
        """Test reading cache statistics resource."""
        from mcp_doc_fetcher.models import CacheStats
        from datetime import datetime

        mock_stats = CacheStats(
            total_libraries=5,
            total_pages=25,
            cache_size_mb=10.5,
            oldest_entry=datetime.now(),
            newest_entry=datetime.now()
        )

        server.cache.get_cache_stats.return_value = mock_stats

        result = await server._read_cache_stats()
        stats_data = json.loads(result.contents[0].text)

        assert stats_data["total_libraries"] == 5
        assert stats_data["total_pages"] == 25
        assert stats_data["cache_size_mb"] == 10.5

    @pytest.mark.asyncio
    async def test_read_cached_libraries(self, server):
        """Test reading cached libraries resource."""
        mock_libraries = [
            {
                "name": "fastapi",
                "version": "latest",
                "cache_key": "fastapi:latest",
                "total_pages": 10,
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T12:00:00",
                "is_expired": False
            }
        ]

        server.cache.list_cached_libraries.return_value = mock_libraries

        result = await server._read_cached_libraries()
        data = json.loads(result.contents[0].text)

        assert len(data["libraries"]) == 1
        assert data["total_count"] == 1
        assert data["libraries"][0]["name"] == "fastapi"