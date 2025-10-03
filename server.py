"""Main MCP server implementation for documentation fetching and search."""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    ListToolsResult,
    TextContent,
    Tool,
)

try:
    from .utils.code_validator import CodeValidator
except ImportError:
    from utils.code_validator import CodeValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)


class CodeFixerServer:
    """MCP server for language-agnostic code validation and fixing."""

    def __init__(self):
        self.server = Server("code-fixer")
        self.validator = CodeValidator()
        self._register_tool_handlers()
        logger.info("Initialized CodeFixerServer")

    def _register_tool_handlers(self):
        """Register MCP tool handlers."""

        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            return ListToolsResult(
                tools=[
                    Tool(
                        name="validate_and_fix_code",
                        description="Validate and fix code in any supported language.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": "The code to validate and fix."
                                },
                                "language": {
                                    "type": "string",
                                    "description": "The programming language (e.g., 'python', 'javascript')."
                                },
                                "project_description": {
                                    "type": "string",
                                    "description": "Optional project description for context."
                                }
                            },
                            "required": ["code", "language"]
                        }
                    )
                ]
            )

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Tool called: {name} with arguments: {arguments}")
            if name == "validate_and_fix_code":
                result = await self.validator.validate_and_fix_code(
                    code=arguments.get("code", ""),
                    language=arguments.get("language", ""),
                    project_description=arguments.get("project_description", "")
                )
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            else:
                return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}, indent=2))]


async def main():
    """Main server entry point."""
    server = CodeFixerServer()
    logger.info("Starting MCP server with stdio transport")
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            server.server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())