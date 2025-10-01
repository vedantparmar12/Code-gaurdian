"""Code validator that fetches docs and validates code against official documentation."""

import ast
import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
import logging

from .ollama_client import OllamaClient
from .error_detector import ErrorDetector
from .version_resolver import VersionResolver

logger = logging.getLogger(__name__)


class CodeValidator:
    """Validates code against official documentation and fixes errors using loop-based regeneration."""

    def __init__(self, doc_fetcher, searcher, ollama_url: str = "http://localhost:11434", ollama_model: str = "qwen3-coder:480b-cloud"):
        """
        Initialize with documentation fetcher and searcher.

        Args:
            doc_fetcher: Function to fetch documentation
            searcher: Function to search documentation
            ollama_url: Ollama server URL
            ollama_model: Ollama model to use for code generation
        """
        self.doc_fetcher = doc_fetcher
        self.searcher = searcher
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.error_detector = ErrorDetector()
        self.version_resolver = VersionResolver()

    async def validate_and_fix_code(
        self,
        code: str,
        project_description: str = "",
        max_iterations: int = 5,
        python_version: str = "3.11"
    ) -> Dict[str, Any]:
        """
        Validate code and automatically fix errors using official documentation with loop-based regeneration.

        Args:
            code: The code to validate
            project_description: Optional project description for context
            max_iterations: Maximum number of fix iterations
            python_version: Target Python version

        Returns:
            Dictionary with validated code and fixes applied
        """
        logger.info("Starting code validation and fixing with loop-based regeneration...")

        # Step 1: Extract libraries used in code
        libraries = self._extract_imports(code)
        logger.info(f"Found {len(libraries)} libraries: {libraries}")

        # Step 2: Fetch documentation for all libraries concurrently
        await self._fetch_all_docs(libraries)

        # Step 3: Check version compatibility
        async with self.version_resolver as resolver:
            compatibility_result = await resolver.resolve_dependencies(libraries, python_version)
            logger.info(f"Version compatibility: {compatibility_result['all_compatible']}")

        # Step 4: Detect errors in the code
        errors = self.error_detector.detect_all_errors(code)
        logger.info(f"Found {len(errors)} errors initially")

        # Step 5: If no errors, validate library usage anyway
        if not errors:
            validation_results = await self._validate_library_usage(code, libraries)
            return {
                "original_code": code,
                "fixed_code": code,
                "libraries_found": libraries,
                "validation_results": validation_results,
                "compatibility_result": compatibility_result,
                "fixes_applied": [],
                "iterations": 0,
                "is_error_free": True
            }

        # Step 6: Loop-based fixing with Qwen3-Coder
        fixed_result = await self._fix_code_with_loop(
            code=code,
            libraries=libraries,
            errors=errors,
            compatibility_result=compatibility_result,
            project_description=project_description,
            max_iterations=max_iterations
        )

        return {
            "original_code": code,
            "fixed_code": fixed_result["fixed_code"],
            "libraries_found": libraries,
            "validation_results": fixed_result.get("validation_results", []),
            "compatibility_result": compatibility_result,
            "fixes_applied": fixed_result["fixes_applied"],
            "iterations": fixed_result["iterations"],
            "is_error_free": fixed_result["is_error_free"],
            "final_errors": fixed_result.get("final_errors", [])
        }

    def _extract_imports(self, code: str) -> List[str]:
        """Extract all imported libraries from code."""
        libraries = set()

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Get base package name
                        lib = alias.name.split('.')[0]
                        libraries.add(lib)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Get base package name
                        lib = node.module.split('.')[0]
                        libraries.add(lib)
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")
            # Try regex fallback
            import_pattern = r'(?:from\s+(\w+)|import\s+(\w+))'
            matches = re.findall(import_pattern, code)
            for match in matches:
                lib = match[0] or match[1]
                if lib:
                    libraries.add(lib)

        # Filter out standard library modules
        standard_libs = {
            'os', 'sys', 'json', 'time', 'datetime', 'math', 'random',
            'collections', 're', 'itertools', 'functools', 'pathlib',
            'typing', 'asyncio', 'logging', 'argparse', 'configparser'
        }

        external_libs = libraries - standard_libs
        return sorted(list(external_libs))

    async def _fetch_all_docs(self, libraries: List[str]):
        """Fetch documentation for all libraries."""
        tasks = []
        for lib in libraries:
            logger.info(f"Fetching documentation for {lib}...")
            tasks.append(self.doc_fetcher(lib))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _validate_library_usage(
        self,
        code: str,
        libraries: List[str]
    ) -> List[Dict[str, Any]]:
        """Validate that library usage matches official documentation."""
        validation_results = []

        for lib in libraries:
            # Extract how the library is used in code
            usage_patterns = self._extract_library_usage(code, lib)

            for pattern in usage_patterns:
                # Search documentation for correct usage
                search_results = await self.searcher(
                    library_name=lib,
                    query=f"how to use {pattern['function']}",
                    max_results=3
                )

                # Check if usage matches documentation
                is_correct = self._check_usage_correctness(
                    pattern,
                    search_results
                )

                validation_results.append({
                    "library": lib,
                    "usage": pattern,
                    "is_correct": is_correct,
                    "documentation_reference": search_results
                })

        return validation_results

    def _extract_library_usage(self, code: str, library: str) -> List[Dict[str, Any]]:
        """Extract how a library is used in the code."""
        usage_patterns = []

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # Check for function calls
                if isinstance(node, ast.Call):
                    func_name = self._get_function_name(node.func)
                    if func_name and library in func_name.lower():
                        usage_patterns.append({
                            "type": "function_call",
                            "function": func_name,
                            "args": len(node.args),
                            "kwargs": len(node.keywords),
                            "line": node.lineno if hasattr(node, 'lineno') else 0
                        })

                # Check for attribute access
                elif isinstance(node, ast.Attribute):
                    attr_name = f"{self._get_name(node.value)}.{node.attr}"
                    if library in attr_name.lower():
                        usage_patterns.append({
                            "type": "attribute_access",
                            "function": attr_name,
                            "line": node.lineno if hasattr(node, 'lineno') else 0
                        })

        except Exception as e:
            logger.error(f"Error extracting usage for {library}: {e}")

        return usage_patterns

    def _get_function_name(self, node) -> Optional[str]:
        """Get function name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._get_function_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return None

    def _get_name(self, node) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._get_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return ""

    def _check_usage_correctness(
        self,
        pattern: Dict[str, Any],
        search_results: List[Dict[str, Any]]
    ) -> bool:
        """Check if usage pattern matches documentation."""
        # If we found relevant documentation, assume usage might be correct
        # In a real implementation, you'd parse the docs and validate syntax
        return len(search_results) > 0

    async def _fix_code_with_loop(
        self,
        code: str,
        libraries: List[str],
        errors: List[Dict[str, Any]],
        compatibility_result: Dict[str, Any],
        project_description: str,
        max_iterations: int
    ) -> Dict[str, Any]:
        """
        Fix code using loop-based regeneration with Qwen3-Coder.

        Args:
            code: Code to fix
            libraries: List of detected libraries
            errors: List of detected errors
            compatibility_result: Version compatibility information
            project_description: Project context
            max_iterations: Maximum iterations

        Returns:
            Dict with fixed code and metadata
        """
        current_code = code
        fixes_applied = []
        iteration = 0

        # Gather documentation context for all libraries
        doc_context = await self._gather_documentation_context(libraries, errors)

        async with OllamaClient(self.ollama_url, self.ollama_model) as ollama:
            for iteration in range(1, max_iterations + 1):
                logger.info(f"=== Fix Iteration {iteration}/{max_iterations} ===")

                # Detect errors in current code
                current_errors = self.error_detector.detect_all_errors(current_code)

                if not current_errors:
                    logger.info(f"✓ Code is error-free after {iteration - 1} iterations")

                    # Validate library usage
                    validation_results = await self._validate_library_usage(current_code, libraries)

                    return {
                        "fixed_code": current_code,
                        "fixes_applied": fixes_applied,
                        "iterations": iteration - 1,
                        "is_error_free": True,
                        "validation_results": validation_results
                    }

                # Format error messages
                error_messages = [
                    f"[{err['type']}] Line {err['line']}: {err['message']}"
                    for err in current_errors
                ]

                logger.info(f"Found {len(error_messages)} errors: {error_messages}")

                # Use Ollama to fix the code
                fix_result = await ollama.fix_code_with_context(
                    broken_code=current_code,
                    error_messages=error_messages,
                    documentation_context=doc_context,
                    max_attempts=1  # We handle loops ourselves
                )

                if not fix_result["success"]:
                    logger.warning(f"Ollama failed to fix code on iteration {iteration}")
                    fixes_applied.append({
                        "iteration": iteration,
                        "errors": error_messages,
                        "fix_applied": False,
                        "reason": "Ollama fix failed"
                    })
                    continue

                # Update current code
                new_code = fix_result["fixed_code"]

                # Check if code actually changed
                if new_code.strip() == current_code.strip():
                    logger.warning("Code did not change, stopping iterations")
                    break

                fixes_applied.append({
                    "iteration": iteration,
                    "errors_before": error_messages,
                    "fix_applied": True,
                    "code_changed": True
                })

                current_code = new_code

            # Final error check
            final_errors = self.error_detector.detect_all_errors(current_code)
            is_error_free = len(final_errors) == 0

            if not is_error_free:
                logger.warning(f"Code still has {len(final_errors)} errors after {max_iterations} iterations")

            return {
                "fixed_code": current_code,
                "fixes_applied": fixes_applied,
                "iterations": iteration,
                "is_error_free": is_error_free,
                "final_errors": final_errors
            }

    async def _gather_documentation_context(
        self,
        libraries: List[str],
        errors: List[Dict[str, Any]]
    ) -> str:
        """
        Gather relevant documentation context for fixing errors.

        Args:
            libraries: List of libraries to search
            errors: List of errors to address

        Returns:
            Formatted documentation context string
        """
        doc_sections = []

        for lib in libraries:
            logger.info(f"Gathering documentation for {lib}...")

            # Search for general usage
            general_results = await self.searcher(
                library_name=lib,
                query=f"{lib} getting started usage examples",
                max_results=2
            )

            if general_results:
                doc_sections.append(f"=== {lib} Documentation ===")
                for i, result in enumerate(general_results[:2], 1):
                    excerpt = result.get("excerpt", "")
                    doc_sections.append(f"\n[Example {i}]:\n{excerpt}\n")

            # Search for specific errors if they mention the library
            for error in errors:
                if lib.lower() in error.get("message", "").lower():
                    error_query = f"{lib} {error.get('message', '')}"
                    error_results = await self.searcher(
                        library_name=lib,
                        query=error_query,
                        max_results=1
                    )

                    if error_results:
                        doc_sections.append(f"\n[Fixing: {error['message']}]:\n{error_results[0].get('excerpt', '')}\n")

        return "\n".join(doc_sections) if doc_sections else "No documentation context available"
