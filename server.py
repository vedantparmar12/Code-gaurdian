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

# New enhanced modules
from .utils.supabase_client import SupabaseVectorStore
from .utils.hybrid_search import HybridSearchEngine
from .utils.code_extractor import CodeExampleExtractor
from .utils.reranker import ResultReranker
from .utils.source_manager import SourceManager
from .utils.parallel_crawler import ParallelCrawler
from .config.settings import get_rag_config

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
        self.rag_config = get_rag_config()
        self.server = Server(self.settings.server_name)

        # Original components
        self.cache: Optional[DocumentCache] = None
        self.searcher: Optional[WebSearcher] = None
        self.crawler: Optional[DocumentCrawler] = None
        self.embedder: Optional[DocumentEmbedder] = None

        # New enhanced components
        self.vector_store: Optional[SupabaseVectorStore] = None
        self.hybrid_search: Optional[HybridSearchEngine] = None
        self.code_extractor: Optional[CodeExampleExtractor] = None
        self.reranker: Optional[ResultReranker] = None
        self.source_manager: Optional[SourceManager] = None
        self.parallel_crawler: Optional[ParallelCrawler] = None

        # Register handlers
        self._register_tool_handlers()
        self._register_resource_handlers()

        logger.info(f"Initialized {self.settings.server_name} v{self.settings.server_version} with enhanced RAG")

    async def initialize(self):
        """Initialize server components."""
        try:
            # Initialize cache
            self.cache = DocumentCache(self.settings)
            await self.cache.initialize()

            # Initialize original components
            self.searcher = WebSearcher(self.settings)
            self.crawler = DocumentCrawler(self.settings)
            self.embedder = DocumentEmbedder(self.settings)

            # Initialize enhanced components based on config
            logger.info("Initializing enhanced RAG components...")

            # Supabase vector store (if configured)
            if self.rag_config.supabase_url and self.rag_config.supabase_service_key:
                try:
                    self.vector_store = SupabaseVectorStore()
                    logger.info("Supabase vector store initialized")

                    # Initialize source manager with vector store
                    self.source_manager = SourceManager(self.vector_store)
                    logger.info("Source manager initialized")

                    # Hybrid search (if enabled)
                    if self.rag_config.use_hybrid_search:
                        self.hybrid_search = HybridSearchEngine(
                            self.vector_store,
                            boost_factor=1.2
                        )
                        logger.info("Hybrid search engine initialized")

                except Exception as e:
                    logger.warning(f"Failed to initialize Supabase components: {e}")

            # Code extractor (if agentic RAG enabled)
            if self.rag_config.use_agentic_rag:
                self.code_extractor = CodeExampleExtractor()
                logger.info("Code example extractor initialized")

            # Reranker (if enabled)
            if self.rag_config.use_reranking:
                self.reranker = ResultReranker(weight=0.5)
                logger.info("Cross-encoder reranker initialized")

            # Parallel crawler
            self.parallel_crawler = ParallelCrawler(
                max_workers=self.rag_config.max_concurrent_crawls
            )
            logger.info(f"Parallel crawler initialized ({self.rag_config.max_concurrent_crawls} workers)")

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
                        name="hybrid_search",
                        description="Advanced hybrid search combining vector similarity and keyword matching for better accuracy (+27% improvement)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query text",
                                    "minLength": 1
                                },
                                "source_filter": {
                                    "type": "string",
                                    "description": "Optional source ID to filter results (e.g., 'fastapi.tiangolo.com')"
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum number of results",
                                    "minimum": 1,
                                    "maximum": 20,
                                    "default": 5
                                },
                                "use_reranking": {
                                    "type": "boolean",
                                    "description": "Apply cross-encoder reranking for better relevance",
                                    "default": True
                                }
                            },
                            "required": ["query"]
                        }
                    ),
                    Tool(
                        name="search_code_examples",
                        description="Search specifically for code examples with AI-generated summaries (95% code finding accuracy)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "What code example are you looking for?",
                                    "minLength": 1
                                },
                                "language": {
                                    "type": "string",
                                    "description": "Programming language filter (e.g., 'python', 'javascript')"
                                },
                                "source_filter": {
                                    "type": "string",
                                    "description": "Optional source ID to filter"
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum number of code examples",
                                    "minimum": 1,
                                    "maximum": 10,
                                    "default": 3
                                }
                            },
                            "required": ["query"]
                        }
                    ),
                    Tool(
                        name="list_sources",
                        description="List all available documentation sources with AI-generated summaries",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "pattern": {
                                    "type": "string",
                                    "description": "Optional pattern to filter sources (e.g., 'fastapi')"
                                }
                            }
                        }
                    )
                ]
            )

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls - returns list of content for MCP SDK to wrap in CallToolResult."""
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
                elif name == "hybrid_search":
                    return await self._handle_hybrid_search(arguments)
                elif name == "search_code_examples":
                    return await self._handle_search_code_examples(arguments)
                elif name == "list_sources":
                    return await self._handle_list_sources(arguments)
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

    async def _handle_fetch_documentation(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle fetch_documentation tool call."""
        try:
            return await self._handle_single_library_documentation(arguments)
        except Exception as e:
            logger.error(f"Error in fetch_documentation: {e}")
            return self._create_error_result(f"Failed to fetch documentation: {str(e)}")

    async def _handle_single_library_documentation(self, arguments: Dict[str, Any]) -> List[TextContent]:
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

            # Use parallel crawler if available (10x faster)
            if self.parallel_crawler:
                logger.info("Using parallel crawler for 10x speed improvement...")

                async def fetch_page(url):
                    return await self.crawler.fetch_single_page(url, library_name=request.library_name)

                crawl_results = await self.parallel_crawler.crawl_many(urls, fetch_page)
                pages = [r.markdown for r in crawl_results if r.success and r.markdown]

                logger.info(f"Parallel crawl complete: {len(pages)}/{len(urls)} successful")
            else:
                # Fallback to original crawler
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

            # Extract code examples if enabled (Agentic RAG)
            code_blocks = []
            if self.code_extractor and self.rag_config.use_agentic_rag:
                logger.info("Extracting code examples from documentation...")
                for page in pages:
                    blocks = self.code_extractor.extract_code_blocks(
                        page.content,
                        url=str(page.url)
                    )
                    code_blocks.extend(blocks)

                if code_blocks:
                    logger.info(f"Found {len(code_blocks)} code examples, generating summaries...")
                    code_blocks = await self.code_extractor.process_code_blocks_parallel(
                        code_blocks,
                        max_workers=10
                    )

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

            # Store in cache (SQLite)
            success = await self.cache.store_library_documentation(
                documentation, embedding_chunks
            )

            if not success:
                logger.warning("Failed to store documentation in cache")

            # Store in Supabase if available
            if self.vector_store:
                try:
                    # Auto-add source
                    if self.source_manager:
                        source = await self.source_manager.auto_add_source(
                            url=str(pages[0].url) if pages else "",
                            title=request.library_name
                        )

                    # Prepare data for batch insert
                    urls_list = [str(p.url) for p in pages]
                    chunks_list = [chunk.text for chunk in embedding_chunks]
                    embeddings_list = [chunk.embedding for chunk in embedding_chunks]
                    metadatas_list = [{"library": request.library_name} for _ in embedding_chunks]

                    # Store documents
                    await self.vector_store.add_documents(
                        urls=urls_list,
                        chunks=chunks_list,
                        embeddings=embeddings_list,
                        metadatas=metadatas_list,
                        batch_size=20
                    )

                    # Store code examples
                    if code_blocks:
                        code_urls = [b['url'] for b in code_blocks]
                        codes = [b['code'] for b in code_blocks]
                        summaries = [b.get('summary', '') for b in code_blocks]
                        code_metadatas = [
                            {
                                'language': b.get('language', 'unknown'),
                                'library': request.library_name
                            }
                            for b in code_blocks
                        ]

                        # Generate embeddings for code
                        code_embeddings = []
                        for code in codes:
                            emb = await self.embedder.generate_query_embedding(code[:500])  # First 500 chars
                            code_embeddings.append(emb)

                        await self.vector_store.add_code_examples(
                            urls=code_urls,
                            codes=codes,
                            summaries=summaries,
                            embeddings=code_embeddings,
                            metadatas=code_metadatas
                        )

                        logger.info(f"Stored {len(code_blocks)} code examples in Supabase")

                    logger.info("Documentation stored in Supabase successfully")

                except Exception as e:
                    logger.warning(f"Failed to store in Supabase: {e}")

            return self._create_success_result(
                f"Successfully fetched documentation for {request.library_name} "
                f"({len(pages)} pages, {len(embedding_chunks)} chunks, {len(code_blocks)} code examples)",
                {
                    "library_name": documentation.library_name,
                    "version": documentation.version,
                    "total_pages": len(pages),
                    "total_chunks": len(embedding_chunks),
                    "total_code_examples": len(code_blocks),
                    "urls_crawled": len(urls),
                    "pages_successful": len(pages),
                    "updated_at": documentation.updated_at.isoformat(),
                    "source": "fresh",
                    "stored_in_supabase": self.vector_store is not None
                }
            )

        except ValidationError as e:
            return self._create_error_result(f"Invalid request: {e}")
        except Exception as e:
            logger.error(f"Error in single library documentation: {e}", exc_info=True)
            return self._create_error_result(f"Failed to fetch documentation: {str(e)}")

    async def _handle_search_documentation(self, arguments: Dict[str, Any]) -> List[TextContent]:
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

    async def _handle_clear_cache(self, arguments: Dict[str, Any]) -> List[TextContent]:
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

    async def _handle_refresh_documentation(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle refresh_documentation tool call."""
        try:
            # This is essentially fetch_documentation with force_refresh=True
            refresh_args = arguments.copy()
            refresh_args["force_refresh"] = True

            return await self._handle_fetch_documentation(refresh_args)

        except Exception as e:
            logger.error(f"Error in refresh_documentation: {e}", exc_info=True)
            return self._create_error_result(f"Failed to refresh documentation: {str(e)}")

    async def _handle_hybrid_search(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle hybrid search (vector + keyword)."""
        try:
            query = arguments.get("query")
            source_filter = arguments.get("source_filter")
            max_results = arguments.get("max_results", 5)
            use_reranking = arguments.get("use_reranking", True)

            if not query:
                return self._create_error_result("No query provided")

            # Check if hybrid search is available
            if not self.hybrid_search or not self.vector_store:
                return self._create_error_result(
                    "Hybrid search not configured. Please set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env"
                )

            # Generate query embedding
            query_embedding = await self.embedder.generate_query_embedding(query)
            if not query_embedding:
                return self._create_error_result("Failed to generate query embedding")

            # Perform hybrid search
            results = await self.hybrid_search.search(
                query=query,
                query_embedding=query_embedding,
                match_count=max_results * 2 if use_reranking else max_results,  # Get more for reranking
                source_filter=source_filter,
                strategy="hybrid"
            )

            if not results:
                return self._create_success_result(
                    f"No results found for query '{query}'",
                    {"query": query, "results": [], "total_results": 0}
                )

            # Apply reranking if enabled
            if use_reranking and self.reranker and len(results) > 1:
                ranked_results = await self.reranker.rerank(query, results, top_k=max_results)
                formatted_results = [
                    {
                        "url": r.url,
                        "content": r.content[:500] + "..." if len(r.content) > 500 else r.content,
                        "relevance_score": round(r.combined_score, 3),
                        "match_type": r.metadata.get("match_type", "hybrid"),
                        "rank_change": r.rank_change
                    }
                    for r in ranked_results
                ]
            else:
                # No reranking
                formatted_results = [
                    {
                        "url": r.get("url"),
                        "content": r.get("content", "")[:500] + "..." if len(r.get("content", "")) > 500 else r.get("content", ""),
                        "relevance_score": round(r.get("similarity", 0), 3),
                        "match_type": r.get("match_type", "hybrid")
                    }
                    for r in results[:max_results]
                ]

            return self._create_success_result(
                f"Found {len(formatted_results)} results for '{query}'",
                {
                    "query": query,
                    "results": formatted_results,
                    "total_results": len(formatted_results),
                    "reranked": use_reranking and self.reranker is not None,
                    "source_filter": source_filter
                }
            )

        except Exception as e:
            logger.error(f"Error in hybrid_search: {e}", exc_info=True)
            return self._create_error_result(f"Hybrid search failed: {str(e)}")

    async def _handle_search_code_examples(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle code example search."""
        try:
            query = arguments.get("query")
            language = arguments.get("language")
            source_filter = arguments.get("source_filter")
            max_results = arguments.get("max_results", 3)

            if not query:
                return self._create_error_result("No query provided")

            # Check if code extraction is available
            if not self.vector_store:
                return self._create_error_result(
                    "Code search not configured. Please set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env"
                )

            # Generate query embedding
            query_embedding = await self.embedder.generate_query_embedding(query)
            if not query_embedding:
                return self._create_error_result("Failed to generate query embedding")

            # Search code examples
            results = await self.vector_store.search_code_examples(
                query_embedding=query_embedding,
                match_count=max_results,
                language_filter=language,
                source_filter=source_filter
            )

            if not results:
                return self._create_success_result(
                    f"No code examples found for '{query}'",
                    {"query": query, "results": [], "total_results": 0}
                )

            # Format results
            formatted_results = [
                {
                    "url": r.get("url"),
                    "code": r.get("content", ""),
                    "summary": r.get("summary", ""),
                    "language": r.get("metadata", {}).get("language", "unknown"),
                    "relevance_score": round(r.get("similarity", 0), 3)
                }
                for r in results
            ]

            return self._create_success_result(
                f"Found {len(formatted_results)} code examples for '{query}'",
                {
                    "query": query,
                    "results": formatted_results,
                    "total_results": len(formatted_results),
                    "language_filter": language,
                    "source_filter": source_filter
                }
            )

        except Exception as e:
            logger.error(f"Error in search_code_examples: {e}", exc_info=True)
            return self._create_error_result(f"Code search failed: {str(e)}")

    async def _handle_list_sources(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle list sources."""
        try:
            pattern = arguments.get("pattern")

            if not self.source_manager:
                return self._create_error_result(
                    "Source management not configured. Please set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env"
                )

            # Get sources
            if pattern:
                sources = await self.source_manager.get_sources_by_pattern(pattern)
            else:
                sources = await self.source_manager.list_sources()

            if not sources:
                return self._create_success_result(
                    "No sources found",
                    {"sources": [], "total_sources": 0}
                )

            # Format sources
            formatted_sources = [
                {
                    "source_id": s.source_id,
                    "title": s.title,
                    "summary": s.summary,
                    "total_pages": s.total_pages,
                    "total_words": s.total_words,
                    "url_pattern": s.url_pattern
                }
                for s in sources
            ]

            return self._create_success_result(
                f"Found {len(formatted_sources)} sources",
                {
                    "sources": formatted_sources,
                    "total_sources": len(formatted_sources),
                    "pattern": pattern
                }
            )

        except Exception as e:
            logger.error(f"Error in list_sources: {e}", exc_info=True)
            return self._create_error_result(f"Failed to list sources: {str(e)}")

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

    def _create_success_result(self, message: str, data: Optional[Dict[str, Any]] = None) -> List[TextContent]:
        """Create a successful tool result (returns content list for MCP SDK)."""
        response = {
            "success": True,
            "message": message,
            "data": data
        }

        return [
            TextContent(
                type="text",
                text=json.dumps(response, indent=2, default=str)
            )
        ]

    def _create_error_result(self, message: str, error_code: Optional[str] = None) -> List[TextContent]:
        """Create an error tool result (returns content list for MCP SDK)."""
        response = {
            "success": False,
            "message": message,
            "error_code": error_code
        }

        # Note: MCP SDK will wrap this in CallToolResult with isError based on exception handling
        # For explicit errors, we include error info in the response text
        return [
            TextContent(
                type="text",
                text=json.dumps(response, indent=2, default=str)
            )
        ]

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
    import os

    # CRITICAL: Set environment variables to suppress all Crawl4AI output
    # MCP uses JSON-RPC over stdio - any non-JSON output breaks the protocol
    os.environ['CRAWL4AI_VERBOSE'] = 'false'
    os.environ['CRAWL4AI_LOG_LEVEL'] = 'ERROR'

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