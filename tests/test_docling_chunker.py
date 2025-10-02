"""
Unit tests for Docling HybridChunker integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from mcp_doc_fetcher.utils.docling_chunker import (
    DoclingHybridChunker,
    FallbackSemanticChunker,
    create_chunker,
    ChunkResult,
    DOCLING_AVAILABLE
)


class TestDoclingHybridChunker:
    """Test Docling-based chunking."""

    @pytest.fixture
    def sample_markdown(self):
        """Sample markdown for testing."""
        return """
# Introduction

This is a sample documentation page with multiple sections.

## Installation

To install the library, use pip:

```python
pip install example-library
```

## Usage

Here's how to use the library:

```python
from example import ExampleClass

# Create instance
obj = ExampleClass()

# Call methods
result = obj.process()
print(result)
```

### Advanced Features

The library also supports advanced features like async processing.

## API Reference

### ExampleClass

Main class for processing data.

**Methods:**
- `process()`: Process data
- `validate()`: Validate inputs
"""

    @pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Docling not installed")
    @pytest.mark.asyncio
    async def test_docling_chunker_initialization(self):
        """Test Docling chunker initializes correctly."""
        chunker = DoclingHybridChunker(
            max_tokens=512,
            embedding_model="nomic-ai/nomic-embed-text-v1.5",
            merge_peers=True
        )

        assert chunker.max_tokens == 512
        assert chunker.merge_peers == True
        assert chunker.tokenizer is not None
        assert chunker.chunker is not None

    @pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Docling not installed")
    @pytest.mark.asyncio
    async def test_docling_chunking_produces_chunks(self, sample_markdown):
        """Test Docling chunker produces valid chunks."""
        chunker = DoclingHybridChunker(max_tokens=200)

        chunks = await chunker.chunk_markdown(
            markdown=sample_markdown,
            title="Test Documentation",
            url="https://example.com/docs"
        )

        assert len(chunks) > 0
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)
        assert all(chunk.token_count > 0 for chunk in chunks)
        assert all(chunk.token_count <= 200 for chunk in chunks)

    @pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Docling not installed")
    @pytest.mark.asyncio
    async def test_docling_chunks_have_context(self, sample_markdown):
        """Test Docling chunks include heading context."""
        chunker = DoclingHybridChunker(max_tokens=200)

        chunks = await chunker.chunk_markdown(
            markdown=sample_markdown,
            title="Test Documentation",
            url="https://example.com/docs"
        )

        # Check that chunks have context flag
        assert any(chunk.has_context for chunk in chunks)

        # Check metadata
        for chunk in chunks:
            assert "title" in chunk.metadata
            assert "url" in chunk.metadata
            assert "chunk_method" in chunk.metadata
            assert chunk.metadata["chunk_method"] == "docling_hybrid"

    @pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Docling not installed")
    @pytest.mark.asyncio
    async def test_token_counting_accuracy(self):
        """Test token counting is accurate."""
        chunker = DoclingHybridChunker(max_tokens=512)

        text = "This is a test sentence for token counting accuracy."
        token_count = chunker.count_tokens(text)

        # Token count should be reasonable (not zero, not too large)
        assert token_count > 0
        assert token_count < 50  # Should be around 10-12 tokens

    @pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Docling not installed")
    @pytest.mark.asyncio
    async def test_empty_content_handling(self):
        """Test chunker handles empty content."""
        chunker = DoclingHybridChunker(max_tokens=512)

        chunks = await chunker.chunk_markdown(
            markdown="",
            title="Empty Doc",
            url="https://example.com"
        )

        assert len(chunks) == 0

    @pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Docling not installed")
    @pytest.mark.asyncio
    async def test_fallback_on_error(self, sample_markdown):
        """Test chunker falls back to simple method on error."""
        chunker = DoclingHybridChunker(max_tokens=512)

        # Mock _markdown_to_docling_document to raise error
        with patch.object(chunker, '_markdown_to_docling_document', side_effect=Exception("Test error")):
            chunks = await chunker.chunk_markdown(
                markdown=sample_markdown,
                title="Test Doc",
                url="https://example.com"
            )

            # Should still produce chunks via fallback
            assert len(chunks) > 0
            # Check fallback method was used
            assert all(chunk.metadata.get("chunk_method") == "fallback_paragraph" for chunk in chunks)


class TestFallbackSemanticChunker:
    """Test fallback chunker."""

    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """
This is paragraph one with some content.

This is paragraph two with more content that should be chunked separately.

This is paragraph three with even more content to test the chunking logic.
"""

    @pytest.mark.asyncio
    async def test_fallback_chunker_initialization(self):
        """Test fallback chunker initializes."""
        chunker = FallbackSemanticChunker(max_tokens=512)

        assert chunker.max_tokens == 512
        assert chunker.chunk_size == 512 * 4  # 4 chars per token

    @pytest.mark.asyncio
    async def test_fallback_chunking(self, sample_text):
        """Test fallback chunker produces chunks."""
        chunker = FallbackSemanticChunker(max_tokens=50)

        chunks = await chunker.chunk_markdown(
            markdown=sample_text,
            title="Test",
            url="https://example.com"
        )

        assert len(chunks) > 0
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

    @pytest.mark.asyncio
    async def test_fallback_token_estimation(self):
        """Test fallback token estimation."""
        chunker = FallbackSemanticChunker(max_tokens=512)

        text = "This is a test with approximately forty characters."
        token_count = chunker.count_tokens(text)

        # Should estimate roughly 10 tokens (40 chars / 4)
        assert token_count == len(text) // 4


class TestChunkerFactory:
    """Test chunker factory function."""

    @pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Docling not installed")
    def test_create_chunker_with_docling(self):
        """Test factory creates Docling chunker when available."""
        chunker = create_chunker(
            use_docling=True,
            max_tokens=512,
            embedding_model="nomic-ai/nomic-embed-text-v1.5"
        )

        assert isinstance(chunker, DoclingHybridChunker)

    def test_create_chunker_fallback(self):
        """Test factory creates fallback chunker when Docling disabled."""
        chunker = create_chunker(
            use_docling=False,
            max_tokens=512
        )

        assert isinstance(chunker, FallbackSemanticChunker)

    @pytest.mark.skipif(DOCLING_AVAILABLE, reason="Test requires Docling to be unavailable")
    def test_create_chunker_when_docling_unavailable(self):
        """Test factory creates fallback when Docling unavailable."""
        chunker = create_chunker(use_docling=True, max_tokens=512)

        # Should fallback to semantic chunker
        assert isinstance(chunker, FallbackSemanticChunker)


class TestMarkdownToDoclingConversion:
    """Test markdown to DoclingDocument conversion."""

    @pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Docling not installed")
    @pytest.mark.asyncio
    async def test_header_extraction(self):
        """Test headers are extracted correctly."""
        chunker = DoclingHybridChunker(max_tokens=512)

        markdown = """
# Main Header

Some content here.

## Sub Header

More content.
"""

        dl_doc = chunker._markdown_to_docling_document(markdown, "Test")

        # Check document was created
        assert dl_doc is not None
        assert dl_doc.name == "Test"

    @pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Docling not installed")
    @pytest.mark.asyncio
    async def test_paragraph_extraction(self):
        """Test paragraphs are extracted correctly."""
        chunker = DoclingHybridChunker(max_tokens=512)

        markdown = """
This is paragraph one.

This is paragraph two.
"""

        dl_doc = chunker._markdown_to_docling_document(markdown, "Test")

        assert dl_doc is not None


@pytest.mark.asyncio
async def test_integration_with_embeddings():
    """Integration test: chunker works with embeddings module."""
    from mcp_doc_fetcher.utils.docling_chunker import create_chunker

    chunker = create_chunker(use_docling=True, max_tokens=512)

    sample_doc = """
# FastAPI Documentation

FastAPI is a modern web framework for Python.

## Installation

```python
pip install fastapi
```

## Quick Start

Create your first API:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
```
"""

    chunks = await chunker.chunk_markdown(
        markdown=sample_doc,
        title="FastAPI Docs",
        url="https://fastapi.tiangolo.com"
    )

    # Verify chunks are suitable for embeddings
    assert len(chunks) > 0
    assert all(chunk.token_count <= 512 for chunk in chunks)
    assert all(len(chunk.content) > 0 for chunk in chunks)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
