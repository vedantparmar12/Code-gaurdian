"""Simple test to verify Docling integration works."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("SIMPLE MCP TEST")
print("=" * 60)

# Test 1: Import config
print("\n1. Testing config import...")
try:
    # Import from config.py (not config directory)
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", Path(__file__).parent / "config.py")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    settings = config_module.get_settings()
    print(f"OK - Config loaded")
    print(f"   - USE_DOCLING_CHUNKING: {settings.use_docling_chunking}")
    print(f"   - MAX_TOKENS_PER_CHUNK: {settings.max_tokens_per_chunk}")
except Exception as e:
    print(f"FAIL - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Import Docling chunker
print("\n2. Testing Docling chunker...")
try:
    from utils.docling_chunker import DOCLING_AVAILABLE, create_chunker
    print(f"OK - Docling chunker imported")
    print(f"   - DOCLING_AVAILABLE: {DOCLING_AVAILABLE}")

    if not DOCLING_AVAILABLE:
        print("\nNOTE: Docling not installed (will use fallback)")
        print("Install with: pip install docling docling-core transformers torch")
except Exception as e:
    print(f"FAIL - {e}")
    sys.exit(1)

# Test 3: Create chunker
print("\n3. Testing chunker creation...")
try:
    chunker = create_chunker(
        use_docling=settings.use_docling_chunking,
        max_tokens=settings.max_tokens_per_chunk,
        embedding_model=settings.chunking_embedding_model
    )
    print(f"OK - Chunker created: {type(chunker).__name__}")
except Exception as e:
    print(f"FAIL - {e}")
    sys.exit(1)

# Test 4: Test chunking
print("\n4. Testing chunking...")
try:
    import asyncio

    async def test_chunk():
        chunks = await chunker.chunk_markdown(
            markdown="# Test\n\nThis is test content.",
            title="Test",
            url="https://test.com"
        )
        return chunks

    chunks = asyncio.run(test_chunk())
    print(f"OK - Created {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i}: {chunk.token_count} tokens")

except Exception as e:
    print(f"FAIL - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test embeddings integration
print("\n5. Testing embeddings integration...")
try:
    from utils.embeddings import DocumentEmbedder
    embedder = DocumentEmbedder(settings)
    print(f"OK - Embedder initialized")
    print(f"   - Chunker available: {embedder.chunker is not None}")
except Exception as e:
    print(f"FAIL - {e}")
    sys.exit(1)

# Test 6: Test MCP server import
print("\n6. Testing MCP server...")
try:
    from server import DocumentationFetcherServer
    print(f"OK - MCP server class imported")
except Exception as e:
    print(f"FAIL - {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("SUCCESS - All tests passed!")
print("=" * 60)

print("\nNext steps:")
print("1. Install Docling (if not installed):")
print("   pip install docling docling-core transformers torch")
print("\n2. Run full integration tests:")
print("   python test_mcp_integration.py")
print("\n3. Start MCP server:")
print("   python -m mcp_doc_fetcher.server")
print("\n4. Configure Claude Desktop and restart")
