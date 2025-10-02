#!/usr/bin/env python
"""
Quick test script to verify Docling integration.

Run this after installation to verify everything works.
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test all required imports."""
    print("Testing imports...")

    try:
        from mcp_doc_fetcher.config import get_settings
        print("✅ Config imports OK")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

    try:
        from mcp_doc_fetcher.utils.docling_chunker import (
            create_chunker,
            DOCLING_AVAILABLE
        )
        print(f"✅ Docling chunker imports OK (available: {DOCLING_AVAILABLE})")
    except ImportError as e:
        print(f"❌ Docling chunker import failed: {e}")
        return False

    try:
        from mcp_doc_fetcher.utils.embeddings import DocumentEmbedder
        print("✅ Embeddings imports OK")
    except ImportError as e:
        print(f"❌ Embeddings import failed: {e}")
        return False

    return True


async def test_chunker():
    """Test chunker functionality."""
    print("\nTesting chunker...")

    from mcp_doc_fetcher.utils.docling_chunker import create_chunker
    from mcp_doc_fetcher.config import get_settings

    try:
        settings = get_settings()
        chunker = create_chunker(
            use_docling=settings.use_docling_chunking,
            max_tokens=settings.max_tokens_per_chunk,
            embedding_model=settings.chunking_embedding_model
        )

        print(f"✅ Chunker created: {type(chunker).__name__}")

        # Test chunking
        sample = "# Test\n\nThis is a test document.\n\n## Section 1\n\nContent here."

        chunks = await chunker.chunk_markdown(
            markdown=sample,
            title="Test",
            url="test.com"
        )

        print(f"✅ Chunking works: {len(chunks)} chunks created")
        return True

    except Exception as e:
        print(f"❌ Chunker test failed: {e}")
        return False


async def test_config():
    """Test configuration."""
    print("\nTesting configuration...")

    from mcp_doc_fetcher.config import get_settings

    try:
        settings = get_settings()
        print(f"✅ Settings loaded:")
        print(f"   - USE_DOCLING_CHUNKING: {settings.use_docling_chunking}")
        print(f"   - MAX_TOKENS_PER_CHUNK: {settings.max_tokens_per_chunk}")
        print(f"   - CHUNKING_EMBEDDING_MODEL: {settings.chunking_embedding_model}")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


async def main():
    """Run all quick tests."""
    print("="*60)
    print("QUICK TEST - Docling Integration")
    print("="*60)

    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))

    # Test 2: Config
    results.append(("Config", await test_config()))

    # Test 3: Chunker
    results.append(("Chunker", await test_chunker()))

    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n🎉 All quick tests passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full tests: python test_mcp_integration.py")
        print("3. Start MCP server: python -m mcp_doc_fetcher.server")
        print("4. Configure Claude Desktop")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
