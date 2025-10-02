#!/usr/bin/env python
"""Test script to verify embedding generation fix."""

import asyncio
import sys
sys.path.insert(0, '.')

from mcp_doc_fetcher.utils.embeddings import DocumentEmbedder
from mcp_doc_fetcher.config import get_settings


async def test_embedding_fix():
    """Test that embedding generation works with direct Ollama client."""
    print("Testing embedding generation fix...")

    settings = get_settings()
    embedder = DocumentEmbedder(settings)

    print(f"Using langchain: {embedder.use_langchain}")
    print(f"Embedding model: {settings.ollama_embedding_model}")

    # Test query embedding
    query = "agent system prompt tools dependencies"
    print(f"\nGenerating embedding for query: '{query}'")

    embedding = await embedder.generate_query_embedding(query)

    if embedding:
        print(f"SUCCESS! Generated embedding with {len(embedding)} dimensions")
        print(f"First 5 values: {embedding[:5]}")
        return True
    else:
        print("FAILED to generate embedding")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_embedding_fix())
    sys.exit(0 if success else 1)
