"""Code validator that fetches docs and validates code against official documentation."""

import asyncio
from typing import List, Dict, Any, Optional
import logging

from language_support.base import LanguageSupport
from language_support.python import PythonLanguageSupport
from language_support.javascript import JavaScriptLanguageSupport
from language_support.typescript import TypeScriptLanguageSupport
from language_support.java import JavaLanguageSupport
from language_support.golang import GoLanguageSupport

logger = logging.getLogger(__name__)


class CodeValidator:
    """Validates code against official documentation and fixes errors using a pluggable language architecture."""

    def __init__(self, ollama_url: str = "http://localhost:11434", ollama_model: str = "qwen3-coder:480b-cloud"):
        """
        Initialize the CodeValidator.

        Args:
            ollama_url: Ollama server URL
            ollama_model: Ollama model to use for code generation
        """
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.languages: Dict[str, LanguageSupport] = {}
        self._register_default_languages()

    def _register_default_languages(self):
        """Register the default language support modules."""
        self.register_language(PythonLanguageSupport(self.ollama_url, self.ollama_model))
        self.register_language(JavaScriptLanguageSupport(self.ollama_url, self.ollama_model))
        self.register_language(TypeScriptLanguageSupport(self.ollama_url, self.ollama_model))
        self.register_language(JavaLanguageSupport(self.ollama_url, self.ollama_model))
        self.register_language(GoLanguageSupport(self.ollama_url, self.ollama_model))
        logger.info("Registered 5 language support modules: Python, JavaScript, TypeScript, Java, Go")

    def register_language(self, lang_support: LanguageSupport):
        """Register a language support module."""
        logger.info(f"Registering language support for: {lang_support.name}")
        self.languages[lang_support.name] = lang_support

    def get_language_support(self, language: str) -> Optional[LanguageSupport]:
        """Get the language support module for a given language."""
        return self.languages.get(language)

    async def validate_and_fix_code(
        self,
        code: str,
        language: str,
        project_description: str = "",
        max_iterations: int = 5,
    ) -> Dict[str, Any]:
        """
        Validate and automatically fix code for a given language.

        Args:
            code: The code to validate.
            language: The programming language of the code (e.g., 'python', 'javascript').
            project_description: Optional project description for context.
            max_iterations: Maximum number of fix iterations.

        Returns:
            Dictionary with validated code and fixes applied.
        """
        logger.info(f"Starting validation for '{language}' code with max {max_iterations} iterations.")

        lang_support = self.get_language_support(language)
        if not lang_support:
            return {"error": f"Language '{language}' is not supported."}

        # Step 1: Extract libraries
        import_extractor = lang_support.import_extractor
        libraries = import_extractor.extract(code)
        logger.info(f"Found {len(libraries)} libraries: {libraries}")

        # Step 2: Detect errors
        error_detector = lang_support.error_detector
        errors = error_detector.detect_errors(code)
        logger.info(f"Found {len(errors)} errors initially.")

        if not errors:
            logger.info("No errors detected. Code seems valid.")
            return {
                "original_code": code,
                "fixed_code": code,
                "libraries_found": libraries,
                "fixes_applied": [],
                "iterations": 0,
                "is_error_free": True,
            }

        # Step 3: Loop-based fixing
        fixed_result = await self._fix_code_with_loop(
            code=code,
            lang_support=lang_support,
            libraries=libraries,
            errors=errors,
            max_iterations=max_iterations,
        )

        return {
            "original_code": code,
            "fixed_code": fixed_result["fixed_code"],
            "libraries_found": libraries,
            "fixes_applied": fixed_result["fixes_applied"],
            "iterations": fixed_result["iterations"],
            "is_error_free": fixed_result["is_error_free"],
            "final_errors": fixed_result.get("final_errors", []),
        }

    async def _fix_code_with_loop(
        self,
        code: str,
        lang_support: LanguageSupport,
        libraries: List[str],
        errors: List[Dict[str, Any]],
        max_iterations: int,
    ) -> Dict[str, Any]:
        """Fix code using loop-based regeneration with an LLM."""
        current_code = code
        fixes_applied = []
        iteration = 0

        doc_context = await self._gather_documentation_context(lang_support, libraries, errors)
        code_fixer = lang_support.code_fixer

        for iteration in range(1, max_iterations + 1):
            logger.info(f"=== Fix Iteration {iteration}/{max_iterations} for {lang_support.name} ===")

            current_errors = lang_support.error_detector.detect_errors(current_code)
            if not current_errors:
                logger.info(f"✓ Code is error-free after {iteration - 1} iterations.")
                return {
                    "fixed_code": current_code,
                    "fixes_applied": fixes_applied,
                    "iterations": iteration - 1,
                    "is_error_free": True,
                }

            new_code = await code_fixer.fix_code(current_code, current_errors, doc_context)

            if new_code.strip() == current_code.strip():
                logger.warning("Code did not change, stopping iterations.")
                break

            fixes_applied.append({
                "iteration": iteration,
                "errors_before": [f"L{e['line']}: {e['message']}" for e in current_errors],
                "fix_applied": True,
            })
            current_code = new_code

        final_errors = lang_support.error_detector.detect_errors(current_code)
        is_error_free = not final_errors
        if not is_error_free:
            logger.warning(f"Code still has {len(final_errors)} errors after {max_iterations} iterations.")

        return {
            "fixed_code": current_code,
            "fixes_applied": fixes_applied,
            "iterations": iteration,
            "is_error_free": is_error_free,
            "final_errors": final_errors,
        }

    async def _gather_documentation_context(
        self,
        lang_support: LanguageSupport,
        libraries: List[str],
        errors: List[Dict[str, Any]],
    ) -> str:
        """Gather relevant documentation context for fixing errors."""
        doc_sections = []
        doc_finder = lang_support.doc_finder

        # In a real implementation, you would use the doc_finder to get URLs
        # and then a crawler/searcher to get content.
        # For now, we'll just simulate this.
        for lib in libraries:
            urls = await doc_finder.find_documentation_urls(lib, max_urls=1)
            if urls:
                doc_sections.append(f"=== Documentation for {lib} ({lang_support.name}) ===\nFound at: {urls[0]}\n")

        if not doc_sections:
            return "No documentation context available."
        return "\n".join(doc_sections)
