"""Tests for web search functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import HttpUrl

from config import Settings
from mcp_doc_fetcher.utils.web_search import WebSearcher, SearchResult


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    return Settings(
        ollama_api_key="test_key",
        ollama_url="http://localhost:11434",
        ollama_embedding_model="nomic-embed-text",
        ollama_chat_model="gpt-oss:120b-cloud"
    )


@pytest.fixture
def web_searcher(mock_settings):
    """Create WebSearcher instance for testing."""
    return WebSearcher(mock_settings)


class TestSearchResult:
    """Test SearchResult model."""

    def test_valid_search_result(self):
        """Test creating a valid SearchResult."""
        result = SearchResult(
            title="FastAPI Documentation",
            url="https://fastapi.tiangolo.com/",
            content="FastAPI is a modern web framework for building APIs"
        )

        assert result.title == "FastAPI Documentation"
        assert str(result.url) == "https://fastapi.tiangolo.com/"
        assert "FastAPI is a modern" in result.content

    def test_search_result_validation(self):
        """Test SearchResult validation."""
        with pytest.raises(ValueError):
            SearchResult(
                title="Test",
                url="https://example.com",
                content=""  # Empty content should fail
            )


class TestWebSearcher:
    """Test WebSearcher functionality."""

    def test_build_search_queries(self, web_searcher):
        """Test search query generation."""
        queries = web_searcher._build_search_queries("fastapi", version="0.100.0")

        assert len(queries) >= 2
        assert any("fastapi 0.100.0" in q.lower() for q in queries)
        assert any("documentation" in q.lower() for q in queries)

    def test_build_search_queries_no_version(self, web_searcher):
        """Test search query generation without version."""
        queries = web_searcher._build_search_queries("fastapi", version=None)

        assert len(queries) >= 2
        assert any("fastapi documentation" in q.lower() for q in queries)
        assert any("2024" in q for q in queries)  # Current year queries

    def test_is_valid_documentation_url(self, web_searcher):
        """Test URL validation for documentation."""
        # Valid documentation URLs
        valid_urls = [
            ("https://fastapi.tiangolo.com/", "fastapi"),
            ("https://docs.python.org/3/", "python"),
            ("https://flask.palletsprojects.com/", "flask"),
            ("https://example.readthedocs.io/", "example"),
        ]

        for url, library in valid_urls:
            assert web_searcher._is_valid_documentation_url(url, library)

        # Invalid URLs
        invalid_urls = [
            "https://github.com/tiangolo/fastapi",  # GitHub repo
            "https://stackoverflow.com/questions/tagged/fastapi",
            "https://reddit.com/r/fastapi",
            "https://twitter.com/fastapi",
        ]

        for url in invalid_urls:
            assert not web_searcher._is_valid_documentation_url(url, "fastapi")

    def test_prioritize_official_sources(self, web_searcher):
        """Test prioritization of official documentation sources."""
        urls = [
            "https://example.readthedocs.io/fastapi",
            "https://fastapi.tiangolo.com/",
            "https://docs.fastapi.com/",
            "https://tutorial-site.com/fastapi-guide",
        ]

        prioritized = web_searcher._prioritize_official_sources(urls, "fastapi")

        # Official URLs should come first (either tiangolo or docs.fastapi)
        assert "fastapi.tiangolo.com" in prioritized[0] or "docs.fastapi.com" in prioritized[0]
        assert len(prioritized) == len(urls)

    @pytest.mark.asyncio
    @patch('ollama.web_search')
    async def test_execute_search_success(self, mock_web_search, web_searcher):
        """Test successful search execution."""
        mock_web_search.return_value = {
            'results': [
                {
                    'title': 'FastAPI Documentation',
                    'url': 'https://fastapi.tiangolo.com/',
                    'content': 'FastAPI framework documentation'
                }
            ]
        }

        results = await web_searcher._execute_search("fastapi documentation")

        assert len(results) == 1
        assert results[0].title == 'FastAPI Documentation'
        assert str(results[0].url) == 'https://fastapi.tiangolo.com/'

    @pytest.mark.asyncio
    @patch('ollama.web_search')
    async def test_execute_search_failure(self, mock_web_search, web_searcher):
        """Test search execution with API failure."""
        mock_web_search.side_effect = Exception("API Error")

        results = await web_searcher._execute_search("fastapi documentation")

        assert results == []

    @pytest.mark.asyncio
    @patch('ollama.web_search')
    async def test_search_documentation_urls(self, mock_web_search, web_searcher):
        """Test complete documentation URL search."""
        mock_web_search.return_value = {
            'results': [
                {
                    'title': 'FastAPI - Official Documentation',
                    'url': 'https://fastapi.tiangolo.com/',
                    'content': 'Official FastAPI documentation'
                },
                {
                    'title': 'FastAPI Tutorial',
                    'url': 'https://fastapi.tiangolo.com/tutorial/',
                    'content': 'Getting started with FastAPI'
                }
            ]
        }

        with patch('asyncio.sleep', new_callable=AsyncMock):  # Speed up test
            urls = await web_searcher.search_documentation_urls("fastapi", max_results=5)

        assert len(urls) >= 1
        assert 'https://fastapi.tiangolo.com/' in urls

    @pytest.mark.asyncio
    @patch('ollama.web_fetch')
    async def test_fetch_page_content_success(self, mock_web_fetch, web_searcher):
        """Test successful page content fetching."""
        mock_web_fetch.return_value = {
            'title': 'FastAPI Documentation',
            'content': '# FastAPI\n\nFastAPI is a modern web framework...'
        }

        content = await web_searcher.fetch_page_content("https://fastapi.tiangolo.com/")

        assert content is not None
        assert 'FastAPI Documentation' in content
        assert 'FastAPI is a modern' in content

    @pytest.mark.asyncio
    @patch('ollama.web_fetch')
    async def test_fetch_page_content_failure(self, mock_web_fetch, web_searcher):
        """Test page content fetching with failure."""
        mock_web_fetch.side_effect = Exception("Fetch error")

        content = await web_searcher.fetch_page_content("https://fastapi.tiangolo.com/")

        assert content is None


@pytest.mark.asyncio
@patch('mcp_doc_fetcher.config.get_settings')
async def test_search_library_documentation_convenience(mock_get_settings):
    """Test convenience function for searching library documentation."""
    from mcp_doc_fetcher.utils.web_search import search_library_documentation

    mock_settings = Settings(
        ollama_api_key="test_key",
        ollama_url="http://localhost:11434",
        ollama_embedding_model="nomic-embed-text",
        ollama_chat_model="gpt-oss:120b-cloud"
    )
    mock_get_settings.return_value = mock_settings

    with patch('ollama.web_search') as mock_web_search:
        mock_web_search.return_value = {'results': []}

        with patch('asyncio.sleep', new_callable=AsyncMock):
            urls = await search_library_documentation("fastapi")

        assert isinstance(urls, list)
        assert mock_web_search.called