
import re
import httpx
import subprocess
import tempfile
import os
from typing import List, Dict, Any
from pathlib import Path

from .base import LanguageSupport, ImportExtractor, DocFinder, ErrorDetector, CodeFixer
from utils.ollama_client import OllamaClient


class TypeScriptImportExtractor(ImportExtractor):
    def extract(self, code: str) -> List[str]:
        """Extract all imported libraries from TypeScript/TSX code."""
        libraries = set()

        # Regex patterns for various import styles
        patterns = [
            r"import\s+.*\s+from\s+['\"]([^'\"]+)['\"]",  # import X from 'lib'
            r"import\s+['\"]([^'\"]+)['\"]",  # import 'lib'
            r"require\(['\"]([^'\"]+)['\"]\)",  # require('lib')
            r"import\s*\(['\"]([^'\"]+)['\"]\)",  # dynamic import('lib')
        ]

        for pattern in patterns:
            matches = re.findall(pattern, code)
            for match in matches:
                lib = match if isinstance(match, str) else match[0]
                # Exclude relative paths and Node.js built-ins
                if lib and not lib.startswith('.') and not lib.startswith('/'):
                    # Handle scoped packages
                    if lib.startswith('@'):
                        parts = lib.split('/')[:2]
                        libraries.add('/'.join(parts))
                    else:
                        libraries.add(lib.split('/')[0])

        # Exclude Node.js built-in modules
        builtins = {
            'fs', 'path', 'http', 'https', 'url', 'os', 'crypto', 'stream',
            'util', 'events', 'buffer', 'process', 'child_process', 'cluster',
            'net', 'dns', 'tls', 'dgram', 'readline', 'zlib', 'querystring',
            'assert', 'constants', 'module', 'perf_hooks', 'timers', 'v8', 'vm'
        }

        return sorted(list(libraries - builtins))


class TypeScriptDocFinder(DocFinder):
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def find_documentation_urls(self, library_name: str, max_urls: int = 10) -> List[str]:
        """Find documentation URLs for TypeScript/React libraries from npm."""
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

                # Get TypeScript declaration files info
                types_info = data.get('types') or data.get('typings')
                if types_info:
                    # Package has built-in TypeScript types
                    pass

                # Check for @types package
                if not library_name.startswith('@types/'):
                    types_package = f"@types/{library_name}"
                    urls.append(f"https://www.npmjs.com/package/{types_package}")

            # Add specific documentation sites for popular libraries
            if library_name == 'react':
                urls.insert(0, "https://react.dev/")
            elif library_name == 'next' or library_name == 'nextjs':
                urls.insert(0, "https://nextjs.org/docs")
            elif library_name == 'typescript':
                urls.insert(0, "https://www.typescriptlang.org/docs/")
            elif library_name.startswith('@types/'):
                base_pkg = library_name.replace('@types/', '')
                urls.insert(0, f"https://www.npmjs.com/package/{base_pkg}")

        except Exception as e:
            print(f"Error finding docs for {library_name}: {e}")

        return urls[:max_urls]


class TypeScriptErrorDetector(ErrorDetector):
    def detect_errors(self, code: str) -> List[Dict[str, Any]]:
        """Detect TypeScript errors using tsc compiler."""
        errors = []

        try:
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Determine file extension (.ts or .tsx)
                is_tsx = 'jsx' in code.lower() or '<' in code and '>' in code and 'return' in code
                extension = '.tsx' if is_tsx else '.ts'
                ts_file = temp_path / f"temp{extension}"

                # Write the TypeScript code
                ts_file.write_text(code)

                # Create a minimal tsconfig.json
                tsconfig = {
                    "compilerOptions": {
                        "target": "ES2020",
                        "module": "commonjs",
                        "jsx": "react" if is_tsx else None,
                        "strict": False,
                        "esModuleInterop": True,
                        "skipLibCheck": True,
                        "noEmit": True
                    }
                }
                tsconfig_file = temp_path / "tsconfig.json"
                import json
                tsconfig_file.write_text(json.dumps(tsconfig))

                # Try to compile with tsc
                result = subprocess.run(
                    ['tsc', '--noEmit', str(ts_file)],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode != 0:
                    # Parse tsc error output
                    error_output = result.stdout + result.stderr
                    for line in error_output.split('\n'):
                        # Parse format: "temp.ts(line,col): error TS1234: message"
                        match = re.match(r'.*\((\d+),(\d+)\):\s*error\s*TS\d+:\s*(.*)', line)
                        if match:
                            line_num, col, message = match.groups()
                            errors.append({
                                'type': 'TypeError',
                                'message': message.strip(),
                                'line': int(line_num),
                                'column': int(col)
                            })

        except FileNotFoundError:
            # TypeScript not installed, use basic checks
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
        """Basic syntax checks when TypeScript compiler is not available."""
        errors = []
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

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
            if line.count('[') != line.count(']'):
                errors.append({
                    'type': 'SyntaxError',
                    'message': 'Unmatched brackets',
                    'line': i
                })

            # Check for React-specific issues
            if 'class=' in stripped:
                errors.append({
                    'type': 'Warning',
                    'message': 'Use className instead of class in JSX',
                    'line': i
                })

        return errors


class TypeScriptCodeFixer(CodeFixer):
    def __init__(self, ollama_url: str, ollama_model: str):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    async def fix_code(self, code: str, errors: List[Dict[str, Any]], doc_context: str) -> str:
        """Fix TypeScript/React code using Ollama."""
        async with OllamaClient(self.ollama_url, self.ollama_model) as ollama:
            error_messages = [f"[{err['type']}] Line {err.get('line', 0)}: {err['message']}" for err in errors]

            # Determine if this is React code
            language = "TypeScript"
            if any(indicator in code for indicator in ['jsx', 'tsx', 'React', 'useState', 'useEffect']):
                language = "TypeScript/React"

            fix_result = await ollama.fix_code_with_context(
                broken_code=code,
                error_messages=error_messages,
                documentation_context=doc_context,
                max_attempts=1,
                language=language
            )

            if fix_result["success"]:
                return fix_result["fixed_code"]
            return code


class TypeScriptLanguageSupport(LanguageSupport):
    def __init__(self, ollama_url: str = "http://localhost:11434", ollama_model: str = "qwen3-coder:480b-cloud"):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    @property
    def name(self) -> str:
        return 'typescript'

    @property
    def import_extractor(self) -> ImportExtractor:
        return TypeScriptImportExtractor()

    @property
    def doc_finder(self) -> DocFinder:
        return TypeScriptDocFinder()

    @property
    def error_detector(self) -> ErrorDetector:
        return TypeScriptErrorDetector()

    @property
    def code_fixer(self) -> CodeFixer:
        return TypeScriptCodeFixer(self.ollama_url, self.ollama_model)
