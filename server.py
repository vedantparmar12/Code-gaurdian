"""Main MCP server implementation for documentation fetching and search."""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListResourcesRequest,
    ListResourcesResult,
    ListToolsRequest,
    ListToolsResult,
    ReadResourceRequest,
    ReadResourceResult,
    Resource,
    TextContent,
    TextResourceContents,
    Tool,
)
from pydantic import ValidationError

from .config import get_settings
from .models import (
    LibraryRequest,
    SearchQuery,
    MCPToolResponse,
    CacheStats,
    LibraryDocumentation,
)
from .utils.cache import DocumentCache
from .utils.crawler import DocumentCrawler
from .utils.embeddings import DocumentEmbedder
from .utils.web_search import WebSearcher
from .utils.code_validator import CodeValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)


class DocumentationFetcherServer:
    """MCP server for documentation fetching and semantic search."""

    def __init__(self):
        self.settings = get_settings()
        self.server = Server(self.settings.server_name)
        self.cache: Optional[DocumentCache] = None
        self.searcher: Optional[WebSearcher] = None
        self.crawler: Optional[DocumentCrawler] = None
        self.embedder: Optional[DocumentEmbedder] = None
        self.code_validator: Optional[CodeValidator] = None

        # Register handlers
        self._register_tool_handlers()
        self._register_resource_handlers()

        logger.info(f"Initialized {self.settings.server_name} v{self.settings.server_version}")

    async def initialize(self):
        """Initialize server components."""
        try:
            # Initialize cache
            self.cache = DocumentCache(self.settings)
            await self.cache.initialize()

            # Initialize other components
            self.searcher = WebSearcher(self.settings)
            self.crawler = DocumentCrawler(self.settings)
            self.embedder = DocumentEmbedder(self.settings)

            # Initialize code validator with Ollama settings
            self.code_validator = CodeValidator(
                doc_fetcher=self._fetch_doc_helper,
                searcher=self._search_doc_helper,
                ollama_url=self.settings.ollama_url,
                ollama_model=self.settings.ollama_chat_model
            )

            logger.info("Server components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize server components: {e}")
            raise

    def _register_tool_handlers(self):
        """Register MCP tool handlers."""

        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available MCP tools."""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="fetch_documentation",
                        description="Fetch and cache documentation for a library OR complete project stack using web search and crawling. Supports intelligent project-aware mode that analyzes project requirements, resolves compatible versions, and recursively fetches all relevant documentation.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "library_name": {
                                    "type": "string",
                                    "description": "Name of the library to fetch documentation for (single library mode)",
                                    "minLength": 1,
                                    "maxLength": 100
                                },
                                "project_description": {
                                    "type": "string",
                                    "description": "Natural language project description (project-aware mode). Example: 'Build a REST API backend with FastAPI, PostgreSQL, Redis for caching, and JWT authentication'",
                                    "minLength": 10,
                                    "maxLength": 2000
                                },
                                "version": {
                                    "type": "string",
                                    "description": "Optional version constraint (e.g., '2.0.0') for single library mode",
                                    "pattern": r"^[\d\.]+$"
                                },
                                "python_version": {
                                    "type": "string",
                                    "description": "Target Python version for project-aware mode (e.g., '3.11')",
                                    "pattern": r"^\d+\.\d+[\+\*]?$"
                                },
                                "max_pages": {
                                    "type": "integer",
                                    "description": "Maximum number of pages to crawl per library",
                                    "minimum": 1,
                                    "maximum": 200,
                                    "default": 50
                                },
                                "force_refresh": {
                                    "type": "boolean",
                                    "description": "Force refresh even if cached",
                                    "default": False
                                }
                            }
                        }
                    ),
                    Tool(
                        name="search_documentation",
                        description="Search cached documentation using semantic similarity",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "library_name": {
                                    "type": "string",
                                    "description": "Name of the library to search in"
                                },
                                "query": {
                                    "type": "string",
                                    "description": "Search query text",
                                    "minLength": 1
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum number of results to return",
                                    "minimum": 1,
                                    "maximum": 20,
                                    "default": 5
                                }
                            },
                            "required": ["library_name", "query"]
                        }
                    ),
                    Tool(
                        name="clear_cache",
                        description="Clear cached documentation",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "library_name": {
                                    "type": "string",
                                    "description": "Optional library name to clear (clears all if not provided)"
                                }
                            }
                        }
                    ),
                    Tool(
                        name="refresh_documentation",
                        description="Force refresh documentation for a specific library",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "library_name": {
                                    "type": "string",
                                    "description": "Name of the library to refresh"
                                },
                                "version": {
                                    "type": "string",
                                    "description": "Optional version constraint"
                                }
                            },
                            "required": ["library_name"]
                        }
                    ),
                    Tool(
                        name="validate_and_fix_code",
                        description="Automatically validate code and fix errors using official documentation. Fetches docs for all libraries, checks compatibility, and corrects usage patterns.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": "The Python code to validate and fix",
                                    "minLength": 1
                                },
                                "project_description": {
                                    "type": "string",
                                    "description": "Optional description of the project for context"
                                }
                            },
                            "required": ["code"]
                        }
                    )
                ]
            )

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls."""
            try:
                logger.info(f"Tool called: {name} with arguments: {arguments}")

                if name == "fetch_documentation":
                    return await self._handle_fetch_documentation(arguments)
                elif name == "search_documentation":
                    return await self._handle_search_documentation(arguments)
                elif name == "clear_cache":
                    return await self._handle_clear_cache(arguments)
                elif name == "refresh_documentation":
                    return await self._handle_refresh_documentation(arguments)
                elif name == "validate_and_fix_code":
                    return await self._handle_validate_and_fix_code(arguments)
                else:
                    return self._create_error_result(f"Unknown tool: {name}")

            except Exception as e:
                logger.error(f"Error in tool {name}: {e}", exc_info=True)
                return self._create_error_result(f"Tool execution failed: {str(e)}")

    def _register_resource_handlers(self):
        """Register MCP resource handlers."""

        @self.server.list_resources()
        async def list_resources() -> ListResourcesResult:
            """List available MCP resources."""
            resources = [
                Resource(
                    uri="doc://cache/stats",
                    name="Cache Statistics",
                    description="Current cache statistics and metrics",
                    mimeType="application/json"
                ),
                Resource(
                    uri="doc://cache/list",
                    name="Cached Libraries",
                    description="List of all cached libraries with metadata",
                    mimeType="application/json"
                )
            ]

            # Add resources for cached libraries
            if self.cache:
                try:
                    cached_libs = await self.cache.list_cached_libraries()
                    for lib in cached_libs:
                        resources.append(Resource(
                            uri=f"doc://{lib['name']}/{lib['version']}/index",
                            name=f"{lib['name']} v{lib['version']} Documentation",
                            description=f"Documentation index for {lib['name']} version {lib['version']}",
                            mimeType="application/json"
                        ))
                except Exception as e:
                    logger.error(f"Error listing cached libraries: {e}")

            return ListResourcesResult(resources=resources)

        @self.server.read_resource()
        async def read_resource(uri: str) -> ReadResourceResult:
            """Read resource content."""
            try:
                logger.debug(f"Reading resource: {uri}")

                if uri == "doc://cache/stats":
                    return await self._read_cache_stats()
                elif uri == "doc://cache/list":
                    return await self._read_cached_libraries()
                elif uri.startswith("doc://") and "/index" in uri:
                    return await self._read_library_index(uri)
                else:
                    return ReadResourceResult(
                        contents=[
                            TextContent(
                                type="text",
                                text=f"Unknown resource: {uri}"
                            )
                        ]
                    )

            except Exception as e:
                logger.error(f"Error reading resource {uri}: {e}")
                return ReadResourceResult(
                    contents=[
                        TextContent(
                            type="text",
                            text=f"Error reading resource: {str(e)}"
                        )
                    ]
                )

    async def _handle_fetch_documentation(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle fetch_documentation tool call."""
        try:
            return await self._handle_single_library_documentation(arguments)
        except Exception as e:
            logger.error(f"Error in fetch_documentation: {e}")
            return self._create_error_result(f"Failed to fetch documentation: {str(e)}")

    async def _handle_single_library_documentation(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle single library documentation fetch (original behavior)."""
        try:
            # Validate arguments
            request = LibraryRequest(**arguments)

            # Check cache first (unless force refresh)
            cache_key = f"{request.library_name}:{request.version or 'latest'}"

            if not request.force_refresh and self.cache:
                cached_doc = await self.cache.get_library_documentation(cache_key)
                if cached_doc:
                    logger.info(f"Found cached documentation for {cache_key}")
                    return self._create_success_result(
                        f"Found cached documentation for {request.library_name} "
                        f"({cached_doc.total_pages} pages)",
                        {
                            "library_name": cached_doc.library_name,
                            "version": cached_doc.version,
                            "total_pages": cached_doc.total_pages,
                            "updated_at": cached_doc.updated_at.isoformat(),
                            "source": "cache"
                        }
                    )

            # Fetch new documentation
            logger.info(f"Fetching fresh documentation for {cache_key}")

            # Search for documentation URLs
            urls = await self.searcher.search_documentation_urls(
                library_name=request.library_name,
                version=request.version,
                max_results=request.max_pages
            )

            if not urls:
                return self._create_error_result(
                    f"No documentation URLs found for {request.library_name}"
                )

            logger.info(f"Found {len(urls)} URLs to crawl")

            # Crawl pages concurrently with library-specific intelligence
            pages = await self.crawler.crawl_pages_concurrent(
                urls,
                library_name=request.library_name
            )

            if not pages:
                return self._create_error_result(
                    f"Failed to crawl documentation pages for {request.library_name}"
                )

            logger.info(f"Successfully crawled {len(pages)} pages")

            # Generate embeddings
            embedding_chunks = await self.embedder.generate_embeddings_for_pages(pages)
            logger.info(f"Generated {len(embedding_chunks)} embedding chunks")

            # Create documentation object
            documentation = LibraryDocumentation(
                library_name=request.library_name,
                version=request.version or "latest",
                pages=pages,
                total_pages=len(pages),
                cache_key=cache_key,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

            # Store in cache
            success = await self.cache.store_library_documentation(
                documentation, embedding_chunks
            )

            if not success:
                logger.warning("Failed to store documentation in cache")

            return self._create_success_result(
                f"Successfully fetched documentation for {request.library_name} "
                f"({len(pages)} pages, {len(embedding_chunks)} chunks)",
                {
                    "library_name": documentation.library_name,
                    "version": documentation.version,
                    "total_pages": len(pages),
                    "total_chunks": len(embedding_chunks),
                    "urls_crawled": len(urls),
                    "pages_successful": len(pages),
                    "updated_at": documentation.updated_at.isoformat(),
                    "source": "fresh"
                }
            )

        except ValidationError as e:
            return self._create_error_result(f"Invalid request: {e}")
        except Exception as e:
            logger.error(f"Error in single library documentation: {e}", exc_info=True)
            return self._create_error_result(f"Failed to fetch documentation: {str(e)}")

    async def _handle_search_documentation(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle search_documentation tool call."""
        try:
            # Validate arguments
            query = SearchQuery(**arguments)

            # Generate query embedding
            query_embedding = await self.embedder.generate_query_embedding(query.query)
            if not query_embedding:
                return self._create_error_result("Failed to generate query embedding")

            # Search in cache
            results = await self.cache.search_embeddings(
                library_name=query.library_name,
                query_embedding=query_embedding,
                max_results=query.max_results,
                min_similarity=0.3
            )

            if not results:
                return self._create_success_result(
                    f"No results found for query '{query.query}' in {query.library_name}",
                    {
                        "library_name": query.library_name,
                        "query": query.query,
                        "results": [],
                        "total_results": 0
                    }
                )

            # Format results
            formatted_results = []
            for text_content, page_title, similarity_score in results:
                formatted_results.append({
                    "title": page_title,
                    "excerpt": text_content[:500] + "..." if len(text_content) > 500 else text_content,
                    "relevance_score": round(similarity_score, 3)
                })

            return self._create_success_result(
                f"Found {len(results)} results for '{query.query}' in {query.library_name}",
                {
                    "library_name": query.library_name,
                    "query": query.query,
                    "results": formatted_results,
                    "total_results": len(results)
                }
            )

        except ValidationError as e:
            return self._create_error_result(f"Invalid request: {e}")
        except Exception as e:
            logger.error(f"Error in search_documentation: {e}", exc_info=True)
            return self._create_error_result(f"Failed to search documentation: {str(e)}")

    async def _handle_clear_cache(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle clear_cache tool call."""
        try:
            library_name = arguments.get("library_name")

            cleared_count = await self.cache.clear_cache(library_name)

            if library_name:
                message = f"Cleared cache for {library_name}"
            else:
                message = f"Cleared all cache entries ({cleared_count} libraries)"

            return self._create_success_result(
                message,
                {
                    "cleared_libraries": cleared_count,
                    "library_name": library_name
                }
            )

        except Exception as e:
            logger.error(f"Error in clear_cache: {e}", exc_info=True)
            return self._create_error_result(f"Failed to clear cache: {str(e)}")

    async def _handle_refresh_documentation(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle refresh_documentation tool call."""
        try:
            # This is essentially fetch_documentation with force_refresh=True
            refresh_args = arguments.copy()
            refresh_args["force_refresh"] = True

            return await self._handle_fetch_documentation(refresh_args)

        except Exception as e:
            logger.error(f"Error in refresh_documentation: {e}", exc_info=True)
            return self._create_error_result(f"Failed to refresh documentation: {str(e)}")

    async def _handle_validate_and_fix_code(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle code validation and fixing."""
        try:
            code = arguments.get("code")
            project_description = arguments.get("project_description", "")

            if not code:
                return self._create_error_result("No code provided for validation")

            logger.info("Starting code validation and fixing...")

            # Validate and fix code
            result = await self.code_validator.validate_and_fix_code(
                code,
                project_description
            )

            return self._create_success_result(
                f"Code validation complete. Found {len(result['libraries_found'])} libraries. "
                f"Applied {len(result['fixes_applied'])} fixes. "
                f"Code is {'error-free' if result['is_error_free'] else 'partially fixed'}.",
                {
                    "libraries_found": result["libraries_found"],
                    "fixed_code": result["fixed_code"],
                    "fixes_applied": result["fixes_applied"],
                    "is_error_free": result["is_error_free"],
                    "validation_results": result["validation_results"]
                }
            )

        except Exception as e:
            logger.error(f"Error in validate_and_fix_code: {e}", exc_info=True)
            return self._create_error_result(f"Code validation failed: {str(e)}")

    async def _fetch_doc_helper(self, library_name: str):
        """Helper to fetch documentation for code validator."""
        return await self._handle_fetch_documentation({
            "library_name": library_name,
            "max_pages": 10
        })

    async def _search_doc_helper(self, library_name: str, query: str, max_results: int = 3):
        """Helper to search documentation for code validator."""
        result = await self._handle_search_documentation({
            "library_name": library_name,
            "query": query,
            "max_results": max_results
        })

        # Extract results from MCP response
        if result.content and len(result.content) > 0:
            import json
            data = json.loads(result.content[0].text)
            return data.get("data", {}).get("results", [])
        return []

    async def _read_cache_stats(self) -> ReadResourceResult:
        """Read cache statistics resource."""
        stats = await self.cache.get_cache_stats()

        return ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri="doc://cache/stats",
                    mimeType="application/json",
                    text=json.dumps({
                        "total_libraries": stats.total_libraries,
                        "total_pages": stats.total_pages,
                        "cache_size_mb": round(stats.cache_size_mb, 2),
                        "oldest_entry": stats.oldest_entry.isoformat() if stats.oldest_entry else None,
                        "newest_entry": stats.newest_entry.isoformat() if stats.newest_entry else None,
                        "updated_at": datetime.now().isoformat()
                    }, indent=2)
                )
            ]
        )

    async def _read_cached_libraries(self) -> ReadResourceResult:
        """Read cached libraries resource."""
        libraries = await self.cache.list_cached_libraries()

        return ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri="doc://cache/list",
                    mimeType="application/json",
                    text=json.dumps({
                        "libraries": libraries,
                        "total_count": len(libraries),
                        "updated_at": datetime.now().isoformat()
                    }, indent=2)
                )
            ]
        )

    async def _read_library_index(self, uri: str) -> ReadResourceResult:
        """Read library documentation index."""
        # Parse URI: doc://library_name/version/index
        parts = uri.replace("doc://", "").split("/")
        if len(parts) >= 2:
            library_name = parts[0]
            version = parts[1]
            cache_key = f"{library_name}:{version}"

            documentation = await self.cache.get_library_documentation(cache_key)
            if documentation:
                pages_info = [
                    {
                        "title": page.title,
                        "url": str(page.url),
                        "word_count": page.word_count,
                        "fetched_at": page.fetched_at.isoformat()
                    }
                    for page in documentation.pages
                ]

                return ReadResourceResult(
                    contents=[
                        TextContent(
                            type="text",
                            text=json.dumps({
                                "library_name": documentation.library_name,
                                "version": documentation.version,
                                "total_pages": documentation.total_pages,
                                "created_at": documentation.created_at.isoformat(),
                                "updated_at": documentation.updated_at.isoformat(),
                                "pages": pages_info
                            }, indent=2)
                        )
                    ]
                )

        return ReadResourceResult(
            contents=[
                TextContent(
                    type="text",
                    text=f"Library documentation not found for URI: {uri}"
                )
            ]
        )

    def _create_success_result(self, message: str, data: Optional[Dict[str, Any]] = None) -> CallToolResult:
        """Create a successful tool result."""
        response = {
            "success": True,
            "message": message,
            "data": data
        }

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, default=str)
                )
            ],
            isError=False
        )

    def _create_error_result(self, message: str, error_code: Optional[str] = None) -> CallToolResult:
        """Create an error tool result."""
        response = {
            "success": False,
            "message": message,
            "error_code": error_code
        }

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, default=str)
                )
            ],
            isError=True
        )

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down MCP server...")
        try:
            if self.cache:
                await self.cache.close()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        logger.info("Server shutdown complete")


async def main():
    """Main server entry point."""
    # Create and initialize server
    doc_server = DocumentationFetcherServer()

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        asyncio.create_task(doc_server.shutdown())

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Initialize server components
        await doc_server.initialize()

        logger.info("Starting MCP server with stdio transport")

        # Run server with stdio transport
        async with stdio_server() as (read_stream, write_stream):
            await doc_server.server.run(
                read_stream,
                write_stream,
                doc_server.server.create_initialization_options()
            )

    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await doc_server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())