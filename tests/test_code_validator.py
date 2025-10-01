"""Unit tests for CodeValidator with loop-based regeneration."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from utils.code_validator import CodeValidator
from utils.error_detector import ErrorDetector
from utils.ollama_client import OllamaClient


@pytest.fixture
def mock_doc_fetcher():
    """Mock documentation fetcher."""
    async def fetcher(library_name: str):
        return {
            "success": True,
            "library_name": library_name,
            "pages": 10
        }
    return fetcher


@pytest.fixture
def mock_searcher():
    """Mock documentation searcher."""
    async def searcher(library_name: str, query: str, max_results: int = 3):
        return [
            {
                "title": f"{library_name} Documentation",
                "excerpt": f"Example usage of {library_name}:\n\nimport {library_name}\n{library_name}.initialize()",
                "relevance_score": 0.95
            }
        ]
    return searcher


@pytest.fixture
def code_validator(mock_doc_fetcher, mock_searcher):
    """Create CodeValidator instance for testing."""
    return CodeValidator(
        doc_fetcher=mock_doc_fetcher,
        searcher=mock_searcher,
        ollama_url="http://localhost:11434",
        ollama_model="qwen3-coder:480b-cloud"
    )


class TestExtractImports:
    """Test import extraction."""

    def test_extract_simple_imports(self, code_validator):
        """Test extracting simple import statements."""
        code = """
import fastapi
import pydantic
from typing import List
"""
        libraries = code_validator._extract_imports(code)
        assert "fastapi" in libraries
        assert "pydantic" in libraries
        # typing is standard library, should be filtered
        assert "typing" not in libraries

    def test_extract_from_imports(self, code_validator):
        """Test extracting from-import statements."""
        code = """
from fastapi import FastAPI, Request
from pydantic import BaseModel
"""
        libraries = code_validator._extract_imports(code)
        assert "fastapi" in libraries
        assert "pydantic" in libraries

    def test_extract_dotted_imports(self, code_validator):
        """Test extracting dotted module imports."""
        code = """
import requests.exceptions
from sqlalchemy.orm import Session
"""
        libraries = code_validator._extract_imports(code)
        assert "requests" in libraries
        assert "sqlalchemy" in libraries

    def test_extract_with_syntax_error(self, code_validator):
        """Test extraction with syntax errors (fallback to regex)."""
        code = """
import requests
def broken_function(
    # Missing closing parenthesis
"""
        libraries = code_validator._extract_imports(code)
        assert "requests" in libraries


class TestErrorDetector:
    """Test error detection."""

    def test_detect_syntax_error(self):
        """Test detecting syntax errors."""
        detector = ErrorDetector()
        code = """
def foo(:
    pass
"""
        errors = detector.detect_all_errors(code)
        assert len(errors) > 0
        assert any(err["type"] == "SyntaxError" for err in errors)

    def test_detect_import_error(self):
        """Test detecting import errors."""
        detector = ErrorDetector()
        code = """
import nonexistent_library_12345
"""
        errors = detector.detect_all_errors(code)
        # Should detect import error for non-existent library
        assert any(err["type"] == "ImportError" for err in errors)

    def test_no_errors_in_valid_code(self):
        """Test that valid code has no errors."""
        detector = ErrorDetector()
        code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"
"""
        errors = detector.detect_all_errors(code)
        assert len(errors) == 0


class TestCodeValidatorIntegration:
    """Integration tests for CodeValidator."""

    @pytest.mark.asyncio
    async def test_validate_correct_code(self, code_validator):
        """Test validating already correct code."""
        code = """
def add(a: int, b: int) -> int:
    return a + b
"""
        result = await code_validator.validate_and_fix_code(code)

        assert result["is_error_free"] is True
        assert result["iterations"] == 0
        assert result["fixed_code"] == code

    @pytest.mark.asyncio
    async def test_extract_libraries_from_code(self, code_validator):
        """Test library extraction from code with imports."""
        code = """
import requests
from fastapi import FastAPI

def main():
    pass
"""
        result = await code_validator.validate_and_fix_code(code, max_iterations=1)

        assert "requests" in result["libraries_found"]
        assert "fastapi" in result["libraries_found"]

    @pytest.mark.asyncio
    async def test_detect_syntax_errors(self, code_validator):
        """Test that syntax errors are detected."""
        code = """
def broken(:
    pass
"""
        # Mock Ollama to avoid actual API calls
        with patch.object(OllamaClient, 'fix_code_with_context') as mock_fix:
            mock_fix.return_value = {
                "success": True,
                "fixed_code": "def broken():\n    pass",
                "attempts": 1
            }

            result = await code_validator.validate_and_fix_code(code, max_iterations=1)

            # Should have attempted to fix
            assert len(result["fixes_applied"]) > 0 or not result["is_error_free"]


class TestOllamaClient:
    """Test Ollama client functionality."""

    @pytest.mark.asyncio
    async def test_extract_code_block(self):
        """Test extracting code from markdown blocks."""
        client = OllamaClient("http://localhost:11434", "qwen3-coder:480b-cloud")

        text = """
Here is the fixed code:

```python
def hello():
    print("Hello World")
```
"""
        extracted = await client.extract_code_block(text)
        assert "def hello():" in extracted
        assert "print" in extracted

    @pytest.mark.asyncio
    async def test_extract_code_without_marker(self):
        """Test extracting code without language marker."""
        client = OllamaClient("http://localhost:11434", "qwen3-coder:480b-cloud")

        text = """
```
def hello():
    pass
```
"""
        extracted = await client.extract_code_block(text)
        assert "def hello():" in extracted

    def test_check_syntax_valid(self):
        """Test syntax checking with valid code."""
        client = OllamaClient("http://localhost:11434", "qwen3-coder:480b-cloud")

        code = """
def add(a, b):
    return a + b
"""
        errors = client._check_syntax(code)
        assert len(errors) == 0

    def test_check_syntax_invalid(self):
        """Test syntax checking with invalid code."""
        client = OllamaClient("http://localhost:11434", "qwen3-coder:480b-cloud")

        code = """
def add(a, b:
    return a + b
"""
        errors = client._check_syntax(code)
        assert len(errors) > 0
        assert "SyntaxError" in errors[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
