"""Core Pydantic models for data validation and type safety."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class LibraryRequest(BaseModel):
    """Request model for fetching library documentation."""

    library_name: str = Field(..., min_length=1, max_length=100)
    version: Optional[str] = Field(None, pattern=r'^[\d\.]+$')
    max_pages: int = Field(10, ge=1, le=50)
    force_refresh: bool = Field(False)

    @field_validator('library_name')
    @classmethod
    def validate_library_name(cls, v: str) -> str:
        """Sanitize library name to prevent injection attacks."""
        if not v.replace('-', '').replace('_', '').replace('.', '').isalnum():
            raise ValueError('Library name contains invalid characters')
        return v.lower()


class ProjectRequest(BaseModel):
    """Request model for fetching complete project documentation."""

    project_description: str = Field(..., min_length=10, max_length=2000)
    python_version: Optional[str] = Field(None, pattern=r'^\d+\.\d+[\+\*]?$')
    max_pages_per_library: int = Field(50, ge=10, le=200)
    force_refresh: bool = Field(False)


class DocumentPage(BaseModel):
    """Single documentation page with content and metadata."""

    url: HttpUrl
    title: str
    content: str
    markdown: str
    fetched_at: datetime
    word_count: int

    @field_validator('content', 'markdown')
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Ensure content is not empty."""
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v


class LibraryDocumentation(BaseModel):
    """Complete library documentation package."""

    library_name: str
    version: str
    pages: List[DocumentPage]
    total_pages: int
    cache_key: str
    created_at: datetime
    updated_at: datetime
    # Enhanced fields for project-aware fetching
    python_version_compat: Optional[str] = None
    dependencies: Dict[str, str] = Field(default_factory=dict)

    @field_validator('pages')
    @classmethod
    def validate_pages_not_empty(cls, v: List[DocumentPage]) -> List[DocumentPage]:
        """Ensure at least one page exists."""
        if not v:
            raise ValueError('Documentation must contain at least one page')
        return v


class SearchQuery(BaseModel):
    """Search query for cached documentation."""

    library_name: str
    query: str
    max_results: int = Field(5, ge=1, le=20)

    @field_validator('query')
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        """Ensure query is not empty."""
        if not v.strip():
            raise ValueError('Search query cannot be empty')
        return v.strip()


class SearchResult(BaseModel):
    """Search result with relevance score."""

    page_url: str
    title: str
    excerpt: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class CacheStats(BaseModel):
    """Cache statistics for monitoring and management."""

    total_libraries: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    cache_size_mb: float = Field(ge=0.0)
    oldest_entry: Optional[datetime] = None
    newest_entry: Optional[datetime] = None


class CrawlStatus(str, Enum):
    """Status of documentation crawling operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class CrawlResult(BaseModel):
    """Result of a crawling operation."""

    status: CrawlStatus
    pages_fetched: int = Field(ge=0)
    pages_failed: int = Field(ge=0)
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None


class EmbeddingChunk(BaseModel):
    """Text chunk with embedding vector."""

    page_id: int
    chunk_index: int
    text_content: str
    token_count: int
    embedding_vector: List[float]

    @field_validator('embedding_vector')
    @classmethod
    def validate_embedding_dimension(cls, v: List[float]) -> List[float]:
        """Ensure embedding has consistent dimensions."""
        if len(v) == 0:
            raise ValueError('Embedding vector cannot be empty')
        if len(v) != 768:  # nomic-embed-text dimension
            raise ValueError(f'Embedding must have 768 dimensions, got {len(v)}')
        return v


class ServerConfig(BaseModel):
    """Server configuration with validation."""

    server_name: str = "doc-fetcher-mcp"
    server_version: str = "1.0.0"
    ollama_api_key: str = Field(..., min_length=1)
    ollama_url: str = Field("http://localhost:11434")
    ollama_embedding_model: str = Field("nomic-embed-text")
    ollama_chat_model: str = Field("gpt-oss:120b-cloud")
    cache_db_path: str = Field("./cache/embeddings.db")
    max_concurrent_crawls: int = Field(5, ge=1, le=20)
    crawl_timeout_seconds: int = Field(30, ge=10, le=120)
    cache_max_age_hours: int = Field(24, ge=1, le=168)  # 1 week max

    @field_validator('ollama_url')
    @classmethod
    def validate_ollama_url(cls, v: str) -> str:
        """Ensure Ollama URL is properly formatted."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Ollama URL must start with http:// or https://')
        return v.rstrip('/')


class MCPToolResponse(BaseModel):
    """Standard response format for MCP tools."""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None


class ResourceInfo(BaseModel):
    """Information about an MCP resource."""

    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"