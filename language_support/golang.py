
import re
import httpx
import subprocess
import tempfile
import os
from typing import List, Dict, Any
from pathlib import Path

from .base import LanguageSupport, ImportExtractor, DocFinder, ErrorDetector, CodeFixer
from utils.ollama_client import OllamaClient


class GoImportExtractor(ImportExtractor):
    def extract(self, code: str) -> List[str]:
        """Extract all imported libraries from Go code."""
        libraries = set()

        # Match single import: import "package"
        single_import_pattern = r'import\s+"([^"]+)"'
        single_matches = re.findall(single_import_pattern, code)
        libraries.update(single_matches)

        # Match grouped imports: import ( "package1" "package2" )
        grouped_import_pattern = r'import\s+\((.*?)\)'
        grouped_matches = re.findall(grouped_import_pattern, code, re.DOTALL)
        for group in grouped_matches:
            package_pattern = r'"([^"]+)"'
            packages = re.findall(package_pattern, group)
            libraries.update(packages)

        # Filter out standard library packages (basic heuristic)
        standard_libs = {
            'fmt', 'os', 'io', 'net', 'http', 'time', 'strings', 'strconv',
            'bytes', 'bufio', 'errors', 'sync', 'context', 'encoding/json',
            'encoding/xml', 'log', 'math', 'sort', 'regexp', 'path', 'filepath',
            'crypto', 'hash', 'reflect', 'runtime', 'testing', 'flag'
        }

        # Keep only third-party packages (those with domain names or github paths)
        external_libs = set()
        for lib in libraries:
            # External packages usually contain '/' or '.'
            if ('/' in lib or '.' in lib) and lib not in standard_libs:
                external_libs.add(lib)

        return sorted(list(external_libs))


class GoDocFinder(DocFinder):
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def find_documentation_urls(self, library_name: str, max_urls: int = 10) -> List[str]:
        """Find documentation URLs for a Go library from pkg.go.dev."""
        urls = []

        try:
            # Go packages are usually hosted on pkg.go.dev
            pkg_url = f"https://pkg.go.dev/{library_name}"
            urls.append(pkg_url)

            # If it's a GitHub package, add the GitHub URL
            if 'github.com' in library_name:
                github_url = f"https://{library_name}"
                urls.append(github_url)

                # Add GitHub documentation URL
                github_parts = library_name.replace('github.com/', '').split('/')
                if len(github_parts) >= 2:
                    owner, repo = github_parts[0], github_parts[1]
                    urls.append(f"https://github.com/{owner}/{repo}")
                    urls.append(f"https://github.com/{owner}/{repo}/blob/main/README.md")

            # Try to verify if the package exists
            try:
                response = await self.client.get(pkg_url, timeout=5)
                if response.status_code != 200:
                    # Package might not exist, try with different version
                    urls.append(f"https://pkg.go.dev/{library_name}@latest")
            except:
                pass

        except Exception as e:
            print(f"Error finding docs for {library_name}: {e}")

        return urls[:max_urls]


class GoErrorDetector(ErrorDetector):
    def detect_errors(self, code: str) -> List[Dict[str, Any]]:
        """Detect Go errors using 'go build' or 'go vet'."""
        errors = []

        try:
            # Create a temporary directory for the Go module
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                go_file = temp_path / "main.go"

                # Ensure code has package declaration
                if 'package ' not in code:
                    code = f"package main\n\n{code}"

                # Write the Go code
                go_file.write_text(code)

                # Initialize a Go module
                subprocess.run(
                    ['go', 'mod', 'init', 'temp'],
                    cwd=temp_dir,
                    capture_output=True,
                    timeout=5
                )

                # Try to build/check the code
                result = subprocess.run(
                    ['go', 'build', '-o', 'temp', 'main.go'],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode != 0:
                    # Parse Go compiler errors
                    error_output = result.stderr
                    for line in error_output.split('\n'):
                        # Parse format: "main.go:line:col: error message"
                        match = re.match(r'.*\.go:(\d+):(\d+):\s*(.*)', line)
                        if match:
                            line_num, col, message = match.groups()
                            errors.append({
                                'type': 'CompileError',
                                'message': message.strip(),
                                'line': int(line_num),
                                'column': int(col)
                            })
                        elif line.strip() and not line.startswith('#'):
                            # Generic error message
                            errors.append({
                                'type': 'Error',
                                'message': line.strip(),
                                'line': 0
                            })

        except FileNotFoundError:
            # Go not installed, use basic regex checks
            errors.extend(self._basic_syntax_check(code))
        except subprocess.TimeoutExpired:
            errors.append({
                'type': 'TimeoutError',
                'message': 'Compilation timed out',
                'line': 0
            })
        except Exception as e:
            errors.append({
                'type': 'UnknownError',
                'message': str(e),
                'line': 0
            })

        return errors

    def _basic_syntax_check(self, code: str) -> List[Dict[str, Any]]:
        """Basic syntax checks when Go compiler is not available."""
        errors = []
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Check for unmatched braces/parentheses
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

            # Check for ':=' vs '=' confusion
            if ':=' in stripped and re.match(r'\s*\w+\s*:=', stripped):
                # This is valid short variable declaration
                pass
            elif '=' in stripped and not any(op in stripped for op in ['==', '!=', '<=', '>=', ':=']):
                # Check if it looks like assignment without declaration
                if not re.match(r'\s*(var|const|type|func|package|import)', stripped):
                    # Might be undeclared variable
                    pass

        return errors


class GoCodeFixer(CodeFixer):
    def __init__(self, ollama_url: str, ollama_model: str):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    async def fix_code(self, code: str, errors: List[Dict[str, Any]], doc_context: str) -> str:
        """Fix Go code using Ollama."""
        async with OllamaClient(self.ollama_url, self.ollama_model) as ollama:
            error_messages = [f"[{err['type']}] Line {err.get('line', 0)}: {err['message']}" for err in errors]

            fix_result = await ollama.fix_code_with_context(
                broken_code=code,
                error_messages=error_messages,
                documentation_context=doc_context,
                max_attempts=1,
                language="Go"
            )

            if fix_result["success"]:
                return fix_result["fixed_code"]
            return code


class GoLanguageSupport(LanguageSupport):
    def __init__(self, ollama_url: str = "http://localhost:11434", ollama_model: str = "qwen3-coder:480b-cloud"):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    @property
    def name(self) -> str:
        return 'go'

    @property
    def import_extractor(self) -> ImportExtractor:
        return GoImportExtractor()

    @property
    def doc_finder(self) -> DocFinder:
        return GoDocFinder()

    @property
    def error_detector(self) -> ErrorDetector:
        return GoErrorDetector()

    @property
    def code_fixer(self) -> CodeFixer:
        return GoCodeFixer(self.ollama_url, self.ollama_model)
