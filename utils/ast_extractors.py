"""AST-based import extractors for accurate code parsing (no regex!)."""

import ast
import logging
from typing import List, Set, Dict, Any, Optional
import subprocess
import tempfile
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class PythonASTExtractor:
    """
    Python AST-based import extractor (most accurate).

    Advantages over regex:
    - Ignores comments and strings
    - Handles multi-line imports
    - Extracts full namespace info
    - Detects dynamic imports
    """

    @staticmethod
    def extract_imports(code: str) -> Dict[str, Any]:
        """
        Extract comprehensive import information using AST.

        Returns:
            {
                'packages': ['requests', 'fastapi'],
                'detailed': [
                    {'module': 'requests', 'names': ['get', 'post'], 'alias': None},
                    {'module': 'fastapi', 'names': ['FastAPI'], 'alias': None}
                ],
                'dynamic_imports': ['importlib.import_module("sklearn")']
            }
        """
        packages = set()
        detailed_imports = []
        dynamic_imports = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # Standard imports: import x, import x as y
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        packages.add(module)
                        detailed_imports.append({
                            'type': 'import',
                            'module': alias.name,
                            'alias': alias.asname,
                            'line': node.lineno
                        })

                # From imports: from x import y
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        packages.add(module)
                        detailed_imports.append({
                            'type': 'from',
                            'module': node.module,
                            'names': [alias.name for alias in node.names],
                            'aliases': {alias.name: alias.asname for alias in node.names if alias.asname},
                            'line': node.lineno
                        })

                # Dynamic imports: __import__, importlib.import_module
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == '__import__':
                        if node.args and isinstance(node.args[0], ast.Constant):
                            pkg = node.args[0].value.split('.')[0]
                            packages.add(pkg)
                            dynamic_imports.append(f"__import__('{node.args[0].value}')")

                    elif isinstance(node.func, ast.Attribute):
                        if (hasattr(node.func.value, 'id') and
                            node.func.value.id == 'importlib' and
                            node.func.attr == 'import_module'):
                            if node.args and isinstance(node.args[0], ast.Constant):
                                pkg = node.args[0].value.split('.')[0]
                                packages.add(pkg)
                                dynamic_imports.append(f"importlib.import_module('{node.args[0].value}')")

        except SyntaxError as e:
            logger.warning(f"AST parsing failed: {e}, falling back to regex")
            return None  # Signal to use fallback

        # Filter standard library
        standard_libs = {
            'os', 'sys', 'json', 'time', 'datetime', 'math', 'random',
            'collections', 're', 'itertools', 'functools', 'pathlib',
            'typing', 'asyncio', 'logging', 'argparse', 'configparser',
            'unittest', 'threading', 'multiprocessing', 'subprocess',
            'urllib', 'http', 'socket', 'email', 'xml', 'csv', 'pickle'
        }

        return {
            'packages': sorted(list(packages - standard_libs)),
            'detailed': detailed_imports,
            'dynamic_imports': dynamic_imports
        }


class JavaScriptASTExtractor:
    """
    JavaScript AST extractor using external parser (requires Node.js + esprima/babel).

    If not available, falls back to regex.
    """

    @staticmethod
    def extract_imports(code: str) -> Optional[Dict[str, Any]]:
        """
        Extract imports using Babel parser (most accurate for modern JS/JSX).

        Requires: npm install -g @babel/parser
        """
        try:
            # Create temp file with code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Create Node.js script to parse with Babel
            parser_script = """
            const parser = require('@babel/parser');
            const fs = require('fs');

            const code = fs.readFileSync(process.argv[2], 'utf8');
            const ast = parser.parse(code, {
                sourceType: 'module',
                plugins: ['jsx', 'typescript']
            });

            const imports = [];
            const requires = [];

            function traverse(node) {
                if (node.type === 'ImportDeclaration') {
                    imports.push({
                        source: node.source.value,
                        specifiers: node.specifiers.map(s => ({
                            type: s.type,
                            local: s.local.name,
                            imported: s.imported ? s.imported.name : null
                        }))
                    });
                }

                if (node.type === 'CallExpression' &&
                    node.callee.name === 'require' &&
                    node.arguments[0]?.type === 'StringLiteral') {
                    requires.push(node.arguments[0].value);
                }

                for (let key in node) {
                    if (node[key] && typeof node[key] === 'object') {
                        if (Array.isArray(node[key])) {
                            node[key].forEach(traverse);
                        } else {
                            traverse(node[key]);
                        }
                    }
                }
            }

            traverse(ast);
            console.log(JSON.stringify({ imports, requires }));
            """

            script_file = Path(temp_file).with_suffix('.parser.js')
            script_file.write_text(parser_script)

            # Run parser
            result = subprocess.run(
                ['node', str(script_file), temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)

                # Extract package names
                packages = set()
                for imp in data['imports']:
                    pkg = imp['source'].split('/')[0]
                    if not pkg.startswith('.'):
                        packages.add(pkg)

                for req in data['requires']:
                    pkg = req.split('/')[0]
                    if not pkg.startswith('.'):
                        packages.add(pkg)

                # Filter Node.js builtins
                builtins = {
                    'fs', 'path', 'http', 'https', 'url', 'os', 'crypto', 'stream',
                    'util', 'events', 'buffer', 'process', 'child_process'
                }

                return {
                    'packages': sorted(list(packages - builtins)),
                    'imports': data['imports'],
                    'requires': data['requires']
                }

        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            logger.warning("Babel parser not available, falling back to regex")
            return None  # Signal fallback

        finally:
            # Cleanup
            try:
                Path(temp_file).unlink(missing_ok=True)
                script_file.unlink(missing_ok=True)
            except:
                pass

        return None


class TypeScriptASTExtractor:
    """
    TypeScript AST extractor using TypeScript compiler API.

    Requires: npm install -g typescript
    """

    @staticmethod
    def extract_imports(code: str) -> Optional[Dict[str, Any]]:
        """
        Extract imports using TypeScript compiler API.

        More accurate than regex for:
        - Type imports (import type { ... })
        - Triple-slash directives
        - Dynamic imports
        """
        try:
            # Create parser script
            parser_script = """
            const ts = require('typescript');
            const fs = require('fs');

            const code = fs.readFileSync(process.argv[2], 'utf8');
            const sourceFile = ts.createSourceFile(
                'temp.ts',
                code,
                ts.ScriptTarget.Latest,
                true
            );

            const imports = [];

            function visit(node) {
                if (ts.isImportDeclaration(node)) {
                    const moduleSpecifier = node.moduleSpecifier.text;
                    const importClause = node.importClause;

                    imports.push({
                        type: 'import',
                        module: moduleSpecifier,
                        isTypeOnly: importClause?.isTypeOnly || false,
                        defaultImport: importClause?.name?.text,
                        namedImports: importClause?.namedBindings?.elements?.map(e => e.name.text) || []
                    });
                }

                ts.forEachChild(node, visit);
            }

            visit(sourceFile);
            console.log(JSON.stringify({ imports }));
            """

            with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
                f.write(code)
                temp_file = f.name

            script_file = Path(temp_file).with_suffix('.parser.js')
            script_file.write_text(parser_script)

            result = subprocess.run(
                ['node', str(script_file), temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)

                packages = set()
                for imp in data['imports']:
                    pkg = imp['module']
                    if pkg.startswith('@'):
                        packages.add('/'.join(pkg.split('/')[:2]))
                    else:
                        packages.add(pkg.split('/')[0])

                return {
                    'packages': sorted(list(packages)),
                    'imports': data['imports']
                }

        except Exception:
            logger.warning("TypeScript parser not available")
            return None

        finally:
            try:
                Path(temp_file).unlink(missing_ok=True)
                script_file.unlink(missing_ok=True)
            except:
                pass

        return None


class GoASTExtractor:
    """
    Go AST extractor using go/parser package.

    This is a lightweight alternative - for full AST, you'd use a Go binary.
    """

    @staticmethod
    def extract_imports(code: str) -> Optional[Dict[str, Any]]:
        """
        Extract imports using `go list -json`.

        For a more robust solution, you'd compile a small Go binary that
        uses go/parser and go/ast packages.
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                go_file = temp_path / "main.go"

                # Ensure package declaration
                if 'package ' not in code:
                    code = f"package main\n\n{code}"

                go_file.write_text(code)

                # Simple regex extraction (Go imports are simpler than JS/TS)
                # For production, use a Go binary with go/parser
                import re

                # Single import
                single_pattern = r'import\s+"([^"]+)"'
                packages = set(re.findall(single_pattern, code))

                # Grouped imports
                grouped_pattern = r'import\s+\((.*?)\)'
                groups = re.findall(grouped_pattern, code, re.DOTALL)
                for group in groups:
                    imports = re.findall(r'"([^"]+)"', group)
                    packages.update(imports)

                # Filter standard library
                external = {p for p in packages if '/' in p or '.' in p}

                return {
                    'packages': sorted(list(external))
                }

        except Exception as e:
            logger.warning(f"Go AST extraction failed: {e}")
            return None


# Factory function to get the best extractor for a language
def get_ast_extractor(language: str):
    """Get the appropriate AST extractor for a language."""
    extractors = {
        'python': PythonASTExtractor,
        'javascript': JavaScriptASTExtractor,
        'typescript': TypeScriptASTExtractor,
        'go': GoASTExtractor,
    }
    return extractors.get(language)
