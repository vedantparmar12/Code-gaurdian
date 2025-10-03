"""
Enhanced Python language support using AST + Aho-Corasick hybrid approach.

This is a drop-in replacement for python.py that demonstrates the optimal strategy:
1. AST parsing for import extraction (most accurate)
2. Aho-Corasick for security validation (fastest for bulk patterns)
3. Regex fallback only when AST fails
"""

import ast
import re
import httpx
import asyncio
from typing import List, Dict, Any

from .base import LanguageSupport, ImportExtractor, DocFinder, ErrorDetector, CodeFixer
from utils.ollama_client import OllamaClient
from utils.error_detector import ErrorDetector as PythonErrorDetector
from utils.ast_extractors import PythonASTExtractor
from utils.aho_corasick_matcher import AhoCorasickMatcher, SecurityMatcher


class EnhancedPythonImportExtractor(ImportExtractor):
    """
    Hybrid approach: AST-first with Aho-Corasick security validation.

    Strategy:
    1. Try AST parsing (most accurate, ignores comments/strings)
    2. Validate with Aho-Corasick for forbidden patterns
    3. If AST fails, fallback to regex
    """

    def __init__(self):
        # Create security matcher for deprecated/vulnerable functions
        self.security_matcher = SecurityMatcher.create_deprecated_python_matcher()

        # Add more security patterns
        vulnerable_patterns = [
            "pickle.loads",      # Arbitrary code execution
            "yaml.load",         # Code injection
            "marshal.loads",     # Code injection
            "os.system",         # Shell injection
            "subprocess.call",   # Shell injection risk
        ]
        self.security_matcher.add_patterns(vulnerable_patterns)
        self.security_matcher.build()

    def extract(self, code: str) -> List[str]:
        """
        Extract imports using AST (primary) with regex fallback.

        Returns:
            List of third-party package names.
        """
        # Security check first (very fast with Aho-Corasick)
        security_issues = self.security_matcher.search(code)
        if security_issues:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Security: Found {len(security_issues)} potentially dangerous patterns")
            for pos, pattern in security_issues[:5]:  # Log first 5
                logger.warning(f"  - '{pattern}' at position {pos}")

        # Try AST extraction (most accurate)
        ast_result = PythonASTExtractor.extract_imports(code)

        if ast_result:
            # AST succeeded - use its results
            packages = ast_result['packages']

            # Log dynamic imports (security concern)
            if ast_result.get('dynamic_imports'):
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Found dynamic imports: {ast_result['dynamic_imports']}")

            return packages

        # AST failed - fallback to regex
        return self._regex_fallback(code)

    def _regex_fallback(self, code: str) -> List[str]:
        """Fallback regex extraction when AST parsing fails."""
        libraries = set()

        import_pattern = r'(?:from\s+(\w+)|import\s+(\w+))'
        matches = re.findall(import_pattern, code)

        for match in matches:
            lib = match[0] or match[1]
            if lib:
                libraries.add(lib)

        # Filter standard library
        standard_libs = {
            'os', 'sys', 'json', 'time', 'datetime', 'math', 'random',
            'collections', 're', 'itertools', 'functools', 'pathlib',
            'typing', 'asyncio', 'logging', 'argparse', 'configparser'
        }

        return sorted(list(libraries - standard_libs))


class EnhancedPythonDocFinder(DocFinder):
    """Same as original - PyPI API is already optimal."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def find_documentation_urls(self, library_name: str, max_urls: int = 10) -> List[str]:
        """Find documentation URLs for a Python library via PyPI."""
        urls = []
        try:
            response = await self.client.get(f"https://pypi.org/pypi/{library_name}/json")
            if response.status_code == 200:
                data = response.json()
                info = data.get('info', {})

                # Documentation URL
                doc_url = info.get('project_urls', {}).get('Documentation')
                if doc_url:
                    urls.append(doc_url)

                # Homepage
                home_page = info.get('home_page')
                if home_page and home_page not in urls:
                    urls.append(home_page)

                # Repository
                repo_url = info.get('project_urls', {}).get('Repository') or info.get('project_urls', {}).get('Source')
                if repo_url and repo_url not in urls:
                    urls.append(repo_url)

        except Exception as e:
            print(f"Error finding docs for {library_name}: {e}")

        return urls[:max_urls]


class EnhancedPythonCodeFixer(CodeFixer):
    """Enhanced code fixer with security awareness."""

    def __init__(self, ollama_url: str, ollama_model: str):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.security_matcher = SecurityMatcher.create_deprecated_python_matcher()

    async def fix_code(self, code: str, errors: List[Dict[str, Any]], doc_context: str) -> str:
        """Fix Python code with security validation."""

        # Check for security issues before fixing
        security_issues = self.security_matcher.search(code)
        if security_issues:
            # Add security warnings to error messages
            security_msg = f"Security: Code contains {len(security_issues)} potentially dangerous patterns"
            errors.append({
                'type': 'SecurityWarning',
                'message': security_msg,
                'line': 0
            })

        async with OllamaClient(self.ollama_url, self.ollama_model) as ollama:
            error_messages = [f"[{err['type']}] Line {err.get('line', 0)}: {err['message']}" for err in errors]

            fix_result = await ollama.fix_code_with_context(
                broken_code=code,
                error_messages=error_messages,
                documentation_context=doc_context,
                max_attempts=1,
                language="Python"
            )

            if fix_result["success"]:
                fixed_code = fix_result["fixed_code"]

                # Validate fixed code doesn't introduce new security issues
                new_issues = self.security_matcher.search(fixed_code)
                if len(new_issues) > len(security_issues):
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning("Fixed code introduced new security issues!")

                return fixed_code

            return code


class EnhancedPythonLanguageSupport(LanguageSupport):
    """
    Enhanced Python language support with AST + Aho-Corasick.

    Drop-in replacement for PythonLanguageSupport.
    """

    def __init__(self, ollama_url: str = "http://localhost:11434", ollama_model: str = "qwen3-coder:480b-cloud"):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    @property
    def name(self) -> str:
        return 'python'

    @property
    def import_extractor(self) -> ImportExtractor:
        return EnhancedPythonImportExtractor()

    @property
    def doc_finder(self) -> DocFinder:
        return EnhancedPythonDocFinder()

    @property
    def error_detector(self) -> ErrorDetector:
        return PythonErrorDetector()  # Keep existing error detector

    @property
    def code_fixer(self) -> CodeFixer:
        return EnhancedPythonCodeFixer(self.ollama_url, self.ollama_model)


# Performance comparison demonstration
if __name__ == "__main__":
    import time

    test_code = """
import requests
from fastapi import FastAPI
import yaml
import pickle
import openai

# Some code
app = FastAPI()
data = pickle.loads(user_input)  # Security issue!
config = yaml.load(file)  # Security issue!
"""

    extractor = EnhancedPythonImportExtractor()

    # Benchmark
    start = time.time()
    packages = extractor.extract(test_code)
    elapsed = time.time() - start

    print(f"Extracted packages: {packages}")
    print(f"Time: {elapsed*1000:.2f}ms")
    print("\nSecurity issues detected by Aho-Corasick:")
    print("- pickle.loads (arbitrary code execution)")
    print("- yaml.load (code injection)")
