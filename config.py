"""Configuration management using pydantic-settings."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file from the project root directory
# This ensures environment variables are loaded even when running from Claude Desktop
project_root = Path(__file__).parent.parent
env_file = project_root / "mcp_doc_fetcher" / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    # Try alternate location
    alt_env_file = project_root / ".env"
    if alt_env_file.exists():
        load_dotenv(alt_env_file)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    ollama_api_key: str = Field(..., description="Ollama API key for web search")

    # Ollama Configuration
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model"
    )
    ollama_chat_model: str = Field(
        default="gpt-oss:120b-cloud",
        description="Ollama chat model for tool use and processing"
    )

    # Cache Configuration
    cache_db_path: str = Field(
        default="./cache/embeddings.db",
        description="SQLite database path for cache"
    )
    cache_max_age_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Maximum age of cached documentation in hours"
    )

    # Crawler Configuration
    max_concurrent_crawls: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent page crawls"
    )
    crawl_timeout_seconds: int = Field(
        default=30,
        ge=10,
        le=120,
        description="Timeout for individual page crawls"
    )
    max_pages_per_library: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum pages to crawl per library"
    )

    # Search Configuration
    search_results_per_query: int = Field(
        default=10,
        ge=5,
        le=20,
        description="Number of search results to process per query"
    )

    # Server Configuration
    server_name: str = Field(
        default="doc-fetcher-mcp",
        description="MCP server name"
    )
    server_version: str = Field(
        default="1.0.0",
        description="MCP server version"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug logging"
    )

    # Embedding Configuration
    embedding_chunk_size: int = Field(
        default=1000,
        ge=100,
        le=2000,
        description="Size of text chunks for embedding"
    )
    embedding_chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=500,
        description="Overlap between text chunks"
    )

    # Neo4j Configuration for Graph RAG
    neo4j_uri: str = Field(
        default="neo4j://127.0.0.1:7687",
        description="Neo4j database URI"
    )
    neo4j_username: str = Field(
        default="neo4j",
        description="Neo4j username"
    )
    neo4j_password: str = Field(
        default="password",
        description="Neo4j password"
    )
    neo4j_database: str = Field(
        default="neo4j",
        description="Neo4j database name"
    )

    # Graph RAG Configuration
    graph_rag_enabled: bool = Field(
        default=True,
        description="Enable Graph RAG features"
    )
    graph_max_entities_per_doc: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Maximum entities to extract per document"
    )
    graph_relationship_threshold: float = Field(
        default=0.7,
        ge=0.1,
        le=1.0,
        description="Minimum confidence threshold for relationships"
    )

    class Config:
        """Pydantic configuration."""
        env_file = [".env", "mcp_doc_fetcher/.env"]  # Try multiple locations
        env_file_encoding = "utf-8"
        case_sensitive = False


def load_settings() -> Settings:
    """Load and validate application settings."""
    try:
        settings = Settings()

        # Validate Ollama URL format
        if not settings.ollama_url.startswith(('http://', 'https://')):
            raise ValueError("Ollama URL must start with http:// or https://")

        # Ensure cache directory exists
        cache_dir = os.path.dirname(settings.cache_db_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        return settings

    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {e}") from e


def get_settings() -> Settings:
    """Get cached settings instance."""
    if not hasattr(get_settings, "_settings"):
        get_settings._settings = load_settings()
    return get_settings._settings