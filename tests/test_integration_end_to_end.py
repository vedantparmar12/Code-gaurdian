"""
End-to-end integration test for the complete code validation workflow.

This test demonstrates the full workflow:
1. Claude generates broken code
2. MCP server detects errors
3. Fetches relevant documentation
4. Uses Qwen3-Coder to fix the code iteratively
5. Returns error-free code
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock
from utils.code_validator import CodeValidator


# Example broken code that Claude might generate
BROKEN_CHATBOT_CODE = """
import fastapi
from fastapi import FastAPI

app = FastAPI(

# Missing closing parenthesis

@app.get("/chat")
async def chat(message: str):
    response = process_message(message)  # Undefined function
    return {"response": response}
"""

FIXED_CHATBOT_CODE = """
from fastapi import FastAPI

app = FastAPI()

@app.get("/chat")
async def chat(message: str):
    response = f"Echo: {message}"
    return {"response": response}
"""


class MockOllamaClient:
    """Mock Ollama client for testing without real API calls."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def fix_code_with_context(
        self,
        broken_code: str,
        error_messages: list,
        documentation_context: str,
        max_attempts: int = 1
    ):
        """
        Mock code fixing that simulates Qwen3-Coder behavior.

        In a real scenario, this would call Ollama API.
        """
        # Simulate fixing the code based on error types
        if "SyntaxError" in str(error_messages):
            # Fix syntax errors
            fixed = broken_code.replace("app = FastAPI(", "app = FastAPI()")
            return {
                "success": True,
                "fixed_code": fixed,
                "attempts": 1
            }

        elif "NameError" in str(error_messages) or "is not defined" in str(error_messages):
            # Fix undefined names
            fixed = FIXED_CHATBOT_CODE
            return {
                "success": True,
                "fixed_code": fixed,
                "attempts": 1
            }

        return {
            "success": False,
            "fixed_code": broken_code,
            "attempts": max_attempts
        }


@pytest.fixture
async def doc_fetcher():
    """Mock documentation fetcher."""
    async def fetcher(library_name: str):
        print(f"[Mock] Fetching documentation for {library_name}...")
        await asyncio.sleep(0.1)  # Simulate network delay
        return {
            "success": True,
            "library_name": library_name,
            "pages": 5,
            "message": f"Documentation for {library_name} fetched"
        }
    return fetcher


@pytest.fixture
async def doc_searcher():
    """Mock documentation searcher."""
    async def searcher(library_name: str, query: str, max_results: int = 3):
        print(f"[Mock] Searching {library_name} docs for: {query}")
        await asyncio.sleep(0.05)  # Simulate search delay

        # Return relevant FastAPI documentation
        if library_name == "fastapi":
            return [
                {
                    "title": "FastAPI Quickstart",
                    "excerpt": """
# FastAPI Quickstart

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello World"}
""",
                    "relevance_score": 0.95
                },
                {
                    "title": "FastAPI Path Parameters",
                    "excerpt": """
# Path Parameters

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
""",
                    "relevance_score": 0.85
                }
            ]
        return []

    return searcher


@pytest.mark.asyncio
async def test_end_to_end_code_validation_and_fixing(doc_fetcher, doc_searcher):
    """
    Complete end-to-end test simulating the full workflow.

    Scenario:
    - User asks Claude: "Create a simple chatbot API with FastAPI"
    - Claude generates broken code (missing parenthesis, undefined function)
    - MCP server automatically:
      1. Detects 2 errors (SyntaxError, NameError)
      2. Fetches FastAPI documentation
      3. Uses Qwen3-Coder to fix errors in loop
      4. Returns error-free code
    """
    print("\n" + "="*70)
    print("INTEGRATION TEST: End-to-End Code Validation Workflow")
    print("="*70)

    # Step 1: Create validator with mock Ollama
    print("\n[Step 1] Initializing CodeValidator...")
    validator = CodeValidator(
        doc_fetcher=doc_fetcher,
        searcher=doc_searcher,
        ollama_url="http://localhost:11434",
        ollama_model="qwen3-coder:480b-cloud"
    )

    # Step 2: Simulate Claude generating broken code
    print("\n[Step 2] Claude generated code (with errors):")
    print("-" * 70)
    print(BROKEN_CHATBOT_CODE)
    print("-" * 70)

    # Step 3: Run validation with mocked Ollama
    print("\n[Step 3] Running code validation and auto-fix...")

    with patch('utils.code_validator.OllamaClient', MockOllamaClient):
        result = await validator.validate_and_fix_code(
            code=BROKEN_CHATBOT_CODE,
            project_description="Simple chatbot API with FastAPI",
            max_iterations=3,
            python_version="3.11"
        )

    # Step 4: Verify results
    print("\n[Step 4] Validation Results:")
    print(f"  - Libraries found: {result['libraries_found']}")
    print(f"  - Iterations taken: {result['iterations']}")
    print(f"  - Error-free: {result['is_error_free']}")
    print(f"  - Fixes applied: {len(result['fixes_applied'])}")

    print("\n[Step 5] Fixed Code:")
    print("-" * 70)
    print(result['fixed_code'])
    print("-" * 70)

    # Assertions
    assert "fastapi" in result["libraries_found"], "Should detect FastAPI library"
    assert result["iterations"] >= 1, "Should have fixed code in at least 1 iteration"
    assert result["is_error_free"], "Code should be error-free after fixing"
    assert len(result["fixes_applied"]) > 0, "Should have applied at least one fix"

    # Verify the fixed code is syntactically valid
    import ast
    try:
        ast.parse(result["fixed_code"])
        print("\n✓ Fixed code is syntactically valid!")
    except SyntaxError as e:
        pytest.fail(f"Fixed code still has syntax errors: {e}")

    print("\n" + "="*70)
    print("✓ END-TO-END TEST PASSED")
    print("="*70)


@pytest.mark.asyncio
async def test_workflow_with_no_errors(doc_fetcher, doc_searcher):
    """
    Test workflow when code has no errors.

    Should return immediately without fixing.
    """
    print("\n[Test] Validating already-correct code...")

    validator = CodeValidator(
        doc_fetcher=doc_fetcher,
        searcher=doc_searcher,
        ollama_url="http://localhost:11434",
        ollama_model="qwen3-coder:480b-cloud"
    )

    correct_code = """
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
"""

    with patch('utils.code_validator.OllamaClient', MockOllamaClient):
        result = await validator.validate_and_fix_code(
            code=correct_code,
            max_iterations=3
        )

    assert result["is_error_free"], "Should recognize code is already correct"
    assert result["iterations"] == 0, "Should not iterate on correct code"
    assert result["fixed_code"] == correct_code, "Should not modify correct code"

    print("✓ Correctly handled error-free code")


@pytest.mark.asyncio
async def test_version_compatibility_check(doc_fetcher, doc_searcher):
    """
    Test that version compatibility is checked.
    """
    print("\n[Test] Checking version compatibility...")

    validator = CodeValidator(
        doc_fetcher=doc_fetcher,
        searcher=doc_searcher,
        ollama_url="http://localhost:11434",
        ollama_model="qwen3-coder:480b-cloud"
    )

    code = """
import requests
import fastapi
"""

    with patch('utils.code_validator.OllamaClient', MockOllamaClient):
        result = await validator.validate_and_fix_code(
            code=code,
            python_version="3.11",
            max_iterations=1
        )

    # Should have checked compatibility
    assert "compatibility_result" in result, "Should include compatibility check"
    assert "packages" in result["compatibility_result"], "Should list packages"

    print("✓ Version compatibility checked")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Running Integration Tests")
    print("="*70 + "\n")

    # Run tests
    pytest.main([__file__, "-v", "-s"])
