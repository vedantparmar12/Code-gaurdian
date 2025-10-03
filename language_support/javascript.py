
import re
import httpx
import subprocess
import tempfile
import os
from typing import List, Dict, Any
from pathlib import Path

from .base import LanguageSupport, ImportExtractor, DocFinder, ErrorDetector, CodeFixer
from utils.ollama_client import OllamaClient

class JavaScriptImportExtractor(ImportExtractor):
    def extract(self, code: str) -> List[str]:
        """Extract all imported libraries from JavaScript code."""
        libraries = set()
        # Regex for require('library') and import ... from 'library'
        patterns = [
            r"require\(['\"]([^'\"]+)['\"]\)",  # require('lib')
            r"import\s+.*\s+from\s+['\"]([^'\"]+)['\"]",  # import x from 'lib'
            r"import\s+['\"]([^'\"]+)['\"]",  # import 'lib'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, code)
            for match in matches:
                lib = match if isinstance(match, str) else match[0]
                # Exclude relative paths and Node.js built-ins
                if lib and not lib.startswith('.') and not lib.startswith('/'):
                    # Get root package name (e.g., '@babel/core' or 'express')
                    if lib.startswith('@'):
                        parts = lib.split('/')[:2]
                        libraries.add('/'.join(parts))
                    else:
                        libraries.add(lib.split('/')[0])

        # Exclude Node.js built-in modules
        builtins = {
            'fs', 'path', 'http', 'https', 'url', 'os', 'crypto', 'stream',
            'util', 'events', 'buffer', 'process', 'child_process', 'cluster',
            'net', 'dns', 'tls', 'dgram', 'readline', 'zlib', 'querystring'
        }
        return sorted(list(libraries - builtins))

class JavaScriptDocFinder(DocFinder):
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def find_documentation_urls(self, library_name: str, max_urls: int = 10) -> List[str]:
        """Find documentation URLs for a JavaScript library from npm."""
        urls = []
        try:
            # Fetch from npm registry
            response = await self.client.get(f"https://registry.npmjs.org/{library_name}")
            if response.status_code == 200:
                data = response.json()

                # Get homepage
                homepage = data.get('homepage')
                if homepage and isinstance(homepage, str):
                    urls.append(homepage)

                # Get repository URL
                repo = data.get('repository', {})
                if isinstance(repo, dict):
                    repo_url = repo.get('url', '')
                    if repo_url:
                        # Clean up git URLs
                        repo_url = repo_url.replace('git+', '').replace('.git', '')
                        repo_url = repo_url.replace('git://', 'https://')
                        if repo_url not in urls:
                            urls.append(repo_url)

                # Get bugs URL
                bugs = data.get('bugs', {})
                if isinstance(bugs, dict):
                    bugs_url = bugs.get('url')
                    if bugs_url and bugs_url not in urls:
                        urls.append(bugs_url)

        except Exception as e:
            print(f"Error finding docs for {library_name}: {e}")

        return urls[:max_urls]

class JavaScriptErrorDetector(ErrorDetector):
    def detect_errors(self, code: str) -> List[Dict[str, Any]]:
        """Detect JavaScript errors using Node.js."""
        errors = []

        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Try to parse with Node.js
                result = subprocess.run(
                    ['node', '--check', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode != 0:
                    # Parse Node.js syntax errors
                    error_output = result.stderr
                    for line in error_output.split('\n'):
                        if 'SyntaxError' in line or 'Error' in line:
                            errors.append({
                                'type': 'SyntaxError',
                                'message': line.strip(),
                                'line': self._extract_line_number(line)
                            })

            finally:
                # Clean up temp file
                os.unlink(temp_file)

        except FileNotFoundError:
            # Node.js not installed, use regex-based basic checks
            errors.extend(self._basic_syntax_check(code))
        except subprocess.TimeoutExpired:
            errors.append({
                'type': 'TimeoutError',
                'message': 'Code execution timed out',
                'line': 0
            })
        except Exception as e:
            errors.append({
                'type': 'UnknownError',
                'message': str(e),
                'line': 0
            })

        return errors

    def _extract_line_number(self, error_line: str) -> int:
        """Extract line number from error message."""
        match = re.search(r':(\d+):', error_line)
        return int(match.group(1)) if match else 0

    def _basic_syntax_check(self, code: str) -> List[Dict[str, Any]]:
        """Basic syntax checks when Node.js is not available."""
        errors = []
        lines = code.split('\n')

        # Check for basic syntax issues
        for i, line in enumerate(lines, 1):
            # Check for unmatched brackets
            if line.count('(') != line.count(')'):
                errors.append({
                    'type': 'SyntaxError',
                    'message': 'Unmatched parentheses',
                    'line': i
                })
            if line.count('{') != line.count('}'):
                errors.append({
                    'type': 'SyntaxError',
                    'message': 'Unmatched braces',
                    'line': i
                })

        return errors

class JavaScriptCodeFixer(CodeFixer):
    def __init__(self, ollama_url: str, ollama_model: str):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    async def fix_code(self, code: str, errors: List[Dict[str, Any]], doc_context: str) -> str:
        """Fix JavaScript code using Ollama."""
        async with OllamaClient(self.ollama_url, self.ollama_model) as ollama:
            error_messages = [f"[{err['type']}] Line {err.get('line', 0)}: {err['message']}" for err in errors]

            fix_result = await ollama.fix_code_with_context(
                broken_code=code,
                error_messages=error_messages,
                documentation_context=doc_context,
                max_attempts=1,
                language="JavaScript"
            )

            if fix_result["success"]:
                return fix_result["fixed_code"]
            return code

class JavaScriptLanguageSupport(LanguageSupport):
    def __init__(self, ollama_url: str = "http://localhost:11434", ollama_model: str = "qwen3-coder:480b-cloud"):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    @property
    def name(self) -> str:
        return 'javascript'

    @property
    def import_extractor(self) -> ImportExtractor:
        return JavaScriptImportExtractor()

    @property
    def doc_finder(self) -> DocFinder:
        return JavaScriptDocFinder()

    @property
    def error_detector(self) -> ErrorDetector:
        return JavaScriptErrorDetector()

    @property
    def code_fixer(self) -> CodeFixer:
        return JavaScriptCodeFixer(self.ollama_url, self.ollama_model)
