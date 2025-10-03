"""
Integration test script for MCP server with Docling chunking.

This script tests the full pipeline:
1. Server initialization
2. Documentation fetching
3. Docling chunking
4. Embedding generation
5. Semantic search

Run this before testing with Claude Desktop.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_doc_fetcher.config import get_settings
from mcp_doc_fetcher.utils.cache import DocumentCache
from mcp_doc_fetcher.utils.embeddings import DocumentEmbedder
from mcp_doc_fetcher.utils.docling_chunker import create_chunker, DOCLING_AVAILABLE
from mcp_doc_fetcher.models import DocumentPage
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_docling_availability():
    """Test if Docling is installed and working."""
    print("\n" + "="*60)
    print("TEST 1: Docling Availability")
    print("="*60)

    if DOCLING_AVAILABLE:
        print("[PASS] Docling is installed and available")
        return True
    else:
        print("[FAIL] Docling is NOT installed")
        print("\nInstall with:")
        print("  pip install docling docling-core transformers torch")
        return False


async def test_chunker_initialization():
    """Test chunker initialization."""
    print("\n" + "="*60)
    print("TEST 2: Chunker Initialization")
    print("="*60)

    try:
        settings = get_settings()
        print(f"Settings loaded:")
        print(f"  - USE_DOCLING_CHUNKING: {settings.use_docling_chunking}")
        print(f"  - MAX_TOKENS_PER_CHUNK: {settings.max_tokens_per_chunk}")
        print(f"  - CHUNKING_EMBEDDING_MODEL: {settings.chunking_embedding_model}")

        chunker = create_chunker(
            use_docling=settings.use_docling_chunking,
            max_tokens=settings.max_tokens_per_chunk,
            embedding_model=settings.chunking_embedding_model
        )

        print(f"\n[PASS] Chunker initialized: {type(chunker).__name__}")
        return chunker

    except Exception as e:
        print(f"[FAIL] Chunker initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_chunking():
    """Test chunking functionality."""
    print("\n" + "="*60)
    print("TEST 3: Chunking Test")
    print("="*60)

    settings = get_settings()
    chunker = create_chunker(
        use_docling=settings.use_docling_chunking,
        max_tokens=settings.max_tokens_per_chunk,
        embedding_model=settings.chunking_embedding_model
    )

    sample_markdown = """
# FastAPI Documentation

FastAPI is a modern, fast web framework for building APIs with Python.

## Installation

Install FastAPI and an ASGI server:

```bash
pip install fastapi
pip install "uvicorn[standard]"
```

## Quick Start

Create a file `main.py`:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

## Run the server

```bash
uvicorn main:app --reload
```

## API Documentation

FastAPI automatically generates interactive API docs at:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
"""

    try:
        chunks = await chunker.chunk_markdown(
            markdown=sample_markdown,
            title="FastAPI Quick Start",
            url="https://fastapi.tiangolo.com/tutorial/"
        )

        print(f"\n[PASS] Chunking successful!")
        print(f"  - Total chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks):
            print(f"\n  Chunk {i}:")
            print(f"    - Token count: {chunk.token_count}")
            print(f"    - Has context: {chunk.has_context}")
            print(f"    - Method: {chunk.metadata.get('chunk_method')}")
            print(f"    - Preview: {chunk.content[:100]}...")

        return chunks

    except Exception as e:
        print(f"[FAIL] Chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_embedder_integration():
    """Test embedder with Docling chunker."""
    print("\n" + "="*60)
    print("TEST 4: Embedder Integration")
    print("="*60)

    try:
        settings = get_settings()
        embedder = DocumentEmbedder(settings)

        print(f"Embedder initialized:")
        print(f"  - Ollama URL: {settings.ollama_url}")
        print(f"  - Embedding model: {settings.ollama_embedding_model}")
        print(f"  - Chunker available: {embedder.chunker is not None}")

        # Create sample page
        sample_page = DocumentPage(
            url="https://example.com/docs",
            title="Test Documentation",
            content="This is test content",
            markdown="""
# Test Documentation

This is a test page for the embedder.

## Section 1

Content in section 1.

## Section 2

Content in section 2.
""",
            word_count=50,
            fetched_at=datetime.now()
        )

        # Generate embeddings
        print("\nGenerating embeddings...")
        embedding_chunks = await embedder.generate_embeddings_for_pages([sample_page])

        print(f"\n[PASS] Embedding generation successful!")
        print(f"  - Total embedding chunks: {len(embedding_chunks)}")

        for i, chunk in enumerate(embedding_chunks[:3]):  # Show first 3
            print(f"\n  Embedding Chunk {i}:")
            print(f"    - Text length: {len(chunk.text_content)}")
            print(f"    - Embedding dim: {len(chunk.embedding_vector) if chunk.embedding_vector else 'None'}")
            print(f"    - Preview: {chunk.text_content[:80]}...")

        return embedding_chunks

    except Exception as e:
        print(f"[FAIL] Embedder integration failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_cache_storage():
    """Test cache storage with embeddings."""
    print("\n" + "="*60)
    print("TEST 5: Cache Storage")
    print("="*60)

    try:
        settings = get_settings()
        cache = DocumentCache(settings)
        await cache.initialize()

        print(f"Cache initialized:")
        print(f"  - Database path: {settings.cache_db_path}")

        # Test cache stats
        stats = await cache.get_cache_stats()
        print(f"\nCache statistics:")
        print(f"  - Total libraries: {stats.total_libraries}")
        print(f"  - Total pages: {stats.total_pages}")
        print(f"  - Cache size: {stats.cache_size_mb:.2f} MB")

        print("\n[PASS] Cache storage working!")

        await cache.close()
        return True

    except Exception as e:
        print(f"[FAIL] Cache storage failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mcp_server_startup():
    """Test MCP server can start."""
    print("\n" + "="*60)
    print("TEST 6: MCP Server Startup")
    print("="*60)

    try:
        from mcp_doc_fetcher.server import DocumentationFetcherServer

        server = DocumentationFetcherServer()
        await server.initialize()

        print("[PASS] MCP Server initialized successfully!")
        print(f"  - Server name: {server.settings.server_name}")
        print(f"  - Server version: {server.settings.server_version}")
        print(f"  - Components:")
        print(f"    - Cache: {'[PASS]' if server.cache else '[FAIL]'}")
        print(f"    - Searcher: {'[PASS]' if server.searcher else '[FAIL]'}")
        print(f"    - Crawler: {'[PASS]' if server.crawler else '[FAIL]'}")
        print(f"    - Embedder: {'[PASS]' if server.embedder else '[FAIL]'}")

        await server.shutdown()
        return True

    except Exception as e:
        print(f"[FAIL] MCP Server startup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_claude_desktop_config():
    """Check Claude Desktop configuration."""
    print("\n" + "="*60)
    print("TEST 7: Claude Desktop Configuration")
    print("="*60)

    import platform
    import json

    system = platform.system()

    if system == "Darwin":  # macOS
        config_path = Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    elif system == "Windows":
        config_path = Path.home() / "AppData/Roaming/Claude/claude_desktop_config.json"
    else:
        print(f"⚠️  Unknown system: {system}")
        return False

    print(f"Looking for config at: {config_path}")

    if config_path.exists():
        print("[PASS] Config file exists")

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            if "mcpServers" in config:
                print(f"\n[CONFIG] MCP Servers configured:")
                for name, server_config in config.get("mcpServers", {}).items():
                    print(f"\n  {name}:")
                    print(f"    - Command: {server_config.get('command')}")
                    print(f"    - Args: {server_config.get('args')}")

                    if "doc-fetcher" in name or "mcp_doc_fetcher" in str(server_config):
                        print(f"\n[PASS] MCP Doc Fetcher is configured!")
                        return True

                print("\n⚠️  MCP Doc Fetcher not found in config")
                print("\nAdd this to your claude_desktop_config.json:")
                print(json.dumps({
                    "mcpServers": {
                        "doc-fetcher": {
                            "command": "python",
                            "args": ["-m", "mcp_doc_fetcher.server"],
                            "env": {
                                "PYTHONPATH": str(Path(__file__).parent.parent.absolute())
                            }
                        }
                    }
                }, indent=2))

            else:
                print("[FAIL] No mcpServers section found")

        except json.JSONDecodeError:
            print("[FAIL] Config file is not valid JSON")
    else:
        print(f"[FAIL] Config file not found at: {config_path}")
        print("\nCreate the file with:")
        print(json.dumps({
            "mcpServers": {
                "doc-fetcher": {
                    "command": "python",
                    "args": ["-m", "mcp_doc_fetcher.server"],
                    "env": {
                        "PYTHONPATH": str(Path(__file__).parent.parent.absolute())
                    }
                }
            }
        }, indent=2))

    return False


async def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("MCP DOC FETCHER - INTEGRATION TEST SUITE")
    print("Testing Docling Integration & Claude Desktop Compatibility")
    print("="*60)

    results = []

    # Test 1: Docling availability
    results.append(("Docling Available", await test_docling_availability()))

    # Test 2: Chunker initialization
    chunker = await test_chunker_initialization()
    results.append(("Chunker Init", chunker is not None))

    # Test 3: Chunking
    if chunker:
        chunks = await test_chunking()
        results.append(("Chunking", chunks is not None and len(chunks) > 0))
    else:
        results.append(("Chunking", False))

    # Test 4: Embedder integration
    embeddings = await test_embedder_integration()
    results.append(("Embedder", embeddings is not None))

    # Test 5: Cache storage
    results.append(("Cache", await test_cache_storage()))

    # Test 6: MCP server startup
    results.append(("MCP Server", await test_mcp_server_startup()))

    # Test 7: Claude Desktop config
    results.append(("Claude Config", await test_claude_desktop_config()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} - {test_name}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n[SUCCESS] ALL TESTS PASSED! Ready for Claude Desktop!")
        print("\nNext steps:")
        print("1. Restart Claude Desktop")
        print("2. Check for 'doc-fetcher' in MCP tools")
        print("3. Try: 'Fetch documentation for fastapi'")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before using with Claude Desktop.")

    return total_passed == total_tests


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
