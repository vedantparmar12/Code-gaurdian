
import re
import httpx
import subprocess
import tempfile
import os
from typing import List, Dict, Any
from pathlib import Path

from .base import LanguageSupport, ImportExtractor, DocFinder, ErrorDetector, CodeFixer
from utils.ollama_client import OllamaClient


class JavaImportExtractor(ImportExtractor):
    def extract(self, code: str) -> List[str]:
        """Extract all imported libraries from Java code."""
        libraries = set()

        # Match import statements: import package.name.*;
        import_pattern = r'import\s+([\w.]+)\s*;'
        matches = re.findall(import_pattern, code)

        for match in matches:
            # Skip java.* and javax.* (standard library)
            if not match.startswith('java.') and not match.startswith('javax.'):
                # Get the base package (e.g., org.springframework)
                parts = match.split('.')
                if len(parts) >= 2:
                    # Use first 2-3 parts as library identifier
                    lib_name = '.'.join(parts[:min(3, len(parts))])
                    libraries.add(lib_name)

        return sorted(list(libraries))


class JavaDocFinder(DocFinder):
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def find_documentation_urls(self, library_name: str, max_urls: int = 10) -> List[str]:
        """Find documentation URLs for a Java library from Maven Central."""
        urls = []

        try:
            # Convert package name to artifact search query
            # e.g., org.springframework.boot -> search for "springframework boot"
            search_query = library_name.replace('.', ' ')

            # Search Maven Central
            search_url = f"https://search.maven.org/solrsearch/select?q={search_query}&rows=5&wt=json"
            response = await self.client.get(search_url)

            if response.status_code == 200:
                data = response.json()
                docs = data.get('response', {}).get('docs', [])

                for doc in docs[:3]:
                    group_id = doc.get('g', '')
                    artifact_id = doc.get('a', '')

                    if group_id and artifact_id:
                        # Try common documentation patterns
                        potential_urls = [
                            f"https://{group_id}.github.io/{artifact_id}/",
                            f"https://javadoc.io/doc/{group_id}/{artifact_id}/",
                            f"https://www.javadoc.io/doc/{group_id}/{artifact_id}/latest/index.html",
                            f"https://docs.{group_id}.io/{artifact_id}/",
                        ]

                        for url in potential_urls:
                            if url not in urls:
                                urls.append(url)

            # Add general documentation sites
            if library_name.startswith('org.springframework'):
                urls.insert(0, f"https://docs.spring.io/spring-framework/docs/current/javadoc-api/")
            elif library_name.startswith('com.google'):
                urls.insert(0, f"https://developers.google.com/")

        except Exception as e:
            print(f"Error finding docs for {library_name}: {e}")

        return urls[:max_urls]


class JavaErrorDetector(ErrorDetector):
    def detect_errors(self, code: str) -> List[Dict[str, Any]]:
        """Detect Java errors using javac compiler."""
        errors = []

        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
                # Try to extract class name from code
                class_match = re.search(r'public\s+class\s+(\w+)', code)
                class_name = class_match.group(1) if class_match else 'TempClass'

                # Ensure code has a class declaration
                if 'class ' not in code:
                    code = f"public class {class_name} {{\n{code}\n}}"

                f.write(code)
                temp_file = f.name

            try:
                # Try to compile with javac
                result = subprocess.run(
                    ['javac', '-Xlint:all', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode != 0:
                    # Parse javac error output
                    error_output = result.stderr
                    for line in error_output.split('\n'):
                        if '.java:' in line:
                            # Parse error format: "file.java:line: error: message"
                            match = re.match(r'.*\.java:(\d+):\s*(\w+):\s*(.*)', line)
                            if match:
                                line_num, error_type, message = match.groups()
                                errors.append({
                                    'type': error_type,
                                    'message': message.strip(),
                                    'line': int(line_num)
                                })

            finally:
                # Clean up temp file and compiled class
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                class_file = temp_file.replace('.java', '.class')
                if os.path.exists(class_file):
                    os.unlink(class_file)

        except FileNotFoundError:
            # javac not installed, use basic regex checks
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
        """Basic syntax checks when javac is not available."""
        errors = []
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            # Check for common syntax issues
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
            # Check for missing semicolons (basic heuristic)
            stripped = line.strip()
            if (stripped and not stripped.endswith((';', '{', '}', ')', ','))
                and not stripped.startswith(('//', '/*', '*', 'import', 'package', 'public', 'private', 'protected'))):
                errors.append({
                    'type': 'Warning',
                    'message': 'Possible missing semicolon',
                    'line': i
                })

        return errors


class JavaCodeFixer(CodeFixer):
    def __init__(self, ollama_url: str, ollama_model: str):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    async def fix_code(self, code: str, errors: List[Dict[str, Any]], doc_context: str) -> str:
        """Fix Java code using Ollama."""
        async with OllamaClient(self.ollama_url, self.ollama_model) as ollama:
            error_messages = [f"[{err['type']}] Line {err.get('line', 0)}: {err['message']}" for err in errors]

            fix_result = await ollama.fix_code_with_context(
                broken_code=code,
                error_messages=error_messages,
                documentation_context=doc_context,
                max_attempts=1,
                language="Java"
            )

            if fix_result["success"]:
                return fix_result["fixed_code"]
            return code


class JavaLanguageSupport(LanguageSupport):
    def __init__(self, ollama_url: str = "http://localhost:11434", ollama_model: str = "qwen3-coder:480b-cloud"):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    @property
    def name(self) -> str:
        return 'java'

    @property
    def import_extractor(self) -> ImportExtractor:
        return JavaImportExtractor()

    @property
    def doc_finder(self) -> DocFinder:
        return JavaDocFinder()

    @property
    def error_detector(self) -> ErrorDetector:
        return JavaErrorDetector()

    @property
    def code_fixer(self) -> CodeFixer:
        return JavaCodeFixer(self.ollama_url, self.ollama_model)
