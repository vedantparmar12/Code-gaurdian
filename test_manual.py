"""Manual test script to verify MCP server tool calling."""
import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mcp_doc_fetcher.server import DocumentationFetcherServer


async def test_fetch_tool():
    """Test the fetch_documentation tool."""
    print("Initializing server...")
    server = DocumentationFetcherServer()
    await server.initialize()

    print("\nTesting fetch_documentation tool...")
    result = await server._handle_fetch_documentation({
        "library_name": "requests",
        "max_pages": 5
    })

    print(f"\nResult type: {type(result)}")
    print(f"Result isError: {result.isError}")
    print(f"Content length: {len(result.content)}")
    print(f"Content[0] type: {type(result.content[0])}")
    print(f"\nFirst 500 chars of response:")
    print(result.content[0].text[:500])

    # Try to serialize it
    try:
        serialized = result.model_dump()
        print(f"\nSerialized successfully: {type(serialized)}")
        print(f"Keys: {serialized.keys()}")
    except Exception as e:
        print(f"\nSerialization error: {e}")

    await server.shutdown()
    print("\nTest completed!")


if __name__ == "__main__":
    asyncio.run(test_fetch_tool())
