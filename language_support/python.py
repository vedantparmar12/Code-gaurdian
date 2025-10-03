
import ast
import re
import httpx
import asyncio
from typing import List, Dict, Any

from .base import LanguageSupport, ImportExtractor, DocFinder, ErrorDetector, CodeFixer
from utils.ollama_client import OllamaClient
from utils.error_detector import ErrorDetector as PythonErrorDetectorImpl

class PythonImportExtractor(ImportExtractor):
    def extract(self, code: str) -> List[str]:
        """Extract all imported libraries from Python code."""
        libraries = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        lib = alias.name.split('.')[0]
                        libraries.add(lib)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        lib = node.module.split('.')[0]
                        libraries.add(lib)
        except SyntaxError:
            import_pattern = r'(?:from\s+(\w+)|import\s+(\w+))'
            matches = re.findall(import_pattern, code)
            for match in matches:
                lib = match[0] or match[1]
                if lib:
                    libraries.add(lib)

        standard_libs = {
            'os', 'sys', 'json', 'time', 'datetime', 'math', 'random',
            'collections', 're', 'itertools', 'functools', 'pathlib',
            'typing', 'asyncio', 'logging', 'argparse', 'configparser'
        }
        return sorted(list(libraries - standard_libs))

class PythonDocFinder(DocFinder):
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def find_documentation_urls(self, library_name: str, max_urls: int = 10) -> List[str]:
        """Find documentation URLs for a Python library."""
        # This is a simplified version of the original UniversalDocFinder
        # In a full implementation, we would move all the logic from UniversalDocFinder here
        urls = []
        try:
            response = await self.client.get(f"https://pypi.org/pypi/{library_name}/json")
            if response.status_code == 200:
                data = response.json()
                info = data.get('info', {})
                doc_url = info.get('project_urls', {}).get('Documentation')
                if doc_url:
                    urls.append(doc_url)
                home_page = info.get('home_page')
                if home_page and home_page not in urls:
                    urls.append(home_page)
        except Exception as e:
            print(f"Error finding docs for {library_name}: {e}")
        return urls

class PythonErrorDetectorWrapper(ErrorDetector):
    """Wrapper to adapt PythonErrorDetectorImpl to the base ErrorDetector interface."""

    def detect_errors(self, code: str) -> List[Dict[str, Any]]:
        """Detect errors using the implementation's detect_all_errors method."""
        detector = PythonErrorDetectorImpl()
        return detector.detect_all_errors(code)


class PythonCodeFixer(CodeFixer):
    def __init__(self, ollama_url: str, ollama_model: str):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    async def fix_code(self, code: str, errors: List[Dict[str, Any]], doc_context: str) -> str:
        """Fix Python code using Ollama."""
        async with OllamaClient(self.ollama_url, self.ollama_model) as ollama:
            error_messages = [f"[{err['type']}] Line {err['line']}: {err['message']}" for err in errors]
            fix_result = await ollama.fix_code_with_context(
                broken_code=code,
                error_messages=error_messages,
                documentation_context=doc_context,
                max_attempts=1
            )
            if fix_result["success"]:
                return fix_result["fixed_code"]
            return code

class PythonLanguageSupport(LanguageSupport):
    def __init__(self, ollama_url: str = "http://localhost:11434", ollama_model: str = "qwen3-coder:480b-cloud"):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    @property
    def name(self) -> str:
        return 'python'

    @property
    def import_extractor(self) -> ImportExtractor:
        return PythonImportExtractor()

    @property
    def doc_finder(self) -> DocFinder:
        return PythonDocFinder()

    @property
    def error_detector(self) -> ErrorDetector:
        return PythonErrorDetectorWrapper() # Wrapper for the existing error detector

    @property
    def code_fixer(self) -> CodeFixer:
        return PythonCodeFixer(self.ollama_url, self.ollama_model)
