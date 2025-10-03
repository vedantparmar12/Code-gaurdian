"""Multi-language import/dependency extractor.

Extracts package dependencies from code in various programming languages.
"""

import re
import logging
from typing import List, Set, Dict, Tuple
from dataclasses import dataclass
from utils.multi_language_registry import Language

logger = logging.getLogger(__name__)


@dataclass
class ExtractedDependency:
    """Extracted dependency information."""
    name: str
    language: Language
    import_statement: str
    line_number: int = 0


class MultiLanguageExtractor:
    """Extract dependencies from code in multiple languages."""

    @staticmethod
    def detect_language(code: str, file_extension: str = None) -> Language:
        """Detect programming language from code or file extension."""

        # File extension mapping
        ext_map = {
            '.py': Language.PYTHON,
            '.js': Language.JAVASCRIPT,
            '.jsx': Language.REACT,
            '.ts': Language.TYPESCRIPT,
            '.tsx': Language.REACT,
            '.java': Language.JAVA,
            '.cpp': Language.CPP,
            '.cc': Language.CPP,
            '.cxx': Language.CPP,
            '.go': Language.GO,
            '.rs': Language.RUST,
            '.cs': Language.CSHARP,
        }

        if file_extension:
            lang = ext_map.get(file_extension.lower())
            if lang:
                return lang

        # Heuristic-based detection from code
        code_lower = code.lower()

        # Python indicators
        if 'import ' in code or 'from ' in code:
            if 'def ' in code or '__init__' in code:
                return Language.PYTHON

        # React indicators
        if 'import react' in code_lower or 'from "react"' in code or 'from \'react\'' in code:
            return Language.REACT

        # TypeScript indicators
        if 'interface ' in code and ': ' in code:
            return Language.TYPESCRIPT

        # JavaScript indicators
        if 'const ' in code or 'let ' in code:
            if 'require(' in code or 'import ' in code:
                return Language.JAVASCRIPT

        # Java indicators
        if 'public class' in code or 'package ' in code:
            return Language.JAVA

        # Rust indicators
        if 'fn main()' in code or ('use ' in code and '::' in code):
            return Language.RUST

        # Go indicators
        if 'func ' in code and 'package ' in code:
            return Language.GO

        # C# indicators
        if 'using ' in code and ('namespace ' in code or 'class ' in code):
            return Language.CSHARP

        return Language.JAVASCRIPT  # Default fallback

    @staticmethod
    def extract_dependencies(code: str, language: Language = None) -> List[ExtractedDependency]:
        """Extract dependencies based on language."""

        if language is None:
            language = MultiLanguageExtractor.detect_language(code)

        extractors = {
            Language.PYTHON: MultiLanguageExtractor._extract_python,
            Language.JAVASCRIPT: MultiLanguageExtractor._extract_javascript,
            Language.TYPESCRIPT: MultiLanguageExtractor._extract_typescript,
            Language.REACT: MultiLanguageExtractor._extract_react,
            Language.JAVA: MultiLanguageExtractor._extract_java,
            Language.CPP: MultiLanguageExtractor._extract_cpp,
            Language.GO: MultiLanguageExtractor._extract_go,
            Language.RUST: MultiLanguageExtractor._extract_rust,
            Language.CSHARP: MultiLanguageExtractor._extract_csharp,
        }

        extractor = extractors.get(language, MultiLanguageExtractor._extract_javascript)
        return extractor(code, language)

    @staticmethod
    def _extract_python(code: str, language: Language) -> List[ExtractedDependency]:
        """Extract Python imports."""
        dependencies = []
        lines = code.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()

            # import module
            match = re.match(r'import\s+([a-zA-Z0-9_\.]+)', line)
            if match:
                module = match.group(1).split('.')[0]
                dependencies.append(ExtractedDependency(
                    name=module,
                    language=language,
                    import_statement=line,
                    line_number=i + 1
                ))

            # from module import ...
            match = re.match(r'from\s+([a-zA-Z0-9_\.]+)\s+import', line)
            if match:
                module = match.group(1).split('.')[0]
                dependencies.append(ExtractedDependency(
                    name=module,
                    language=language,
                    import_statement=line,
                    line_number=i + 1
                ))

        return dependencies

    @staticmethod
    def _extract_javascript(code: str, language: Language) -> List[ExtractedDependency]:
        """Extract JavaScript/Node.js imports."""
        dependencies = []
        lines = code.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()

            # import ... from 'package'
            match = re.search(r'import\s+.*?from\s+[\'"]([^\'"\s]+)[\'"]', line)
            if match:
                package = match.group(1)
                # Remove relative paths
                if not package.startswith('.') and not package.startswith('/'):
                    # Extract package name (before any subpath)
                    package_name = package.split('/')[0]
                    dependencies.append(ExtractedDependency(
                        name=package_name,
                        language=language,
                        import_statement=line,
                        line_number=i + 1
                    ))

            # require('package')
            match = re.search(r'require\([\'"]([^\'"\s]+)[\'"]\)', line)
            if match:
                package = match.group(1)
                if not package.startswith('.') and not package.startswith('/'):
                    package_name = package.split('/')[0]
                    dependencies.append(ExtractedDependency(
                        name=package_name,
                        language=language,
                        import_statement=line,
                        line_number=i + 1
                    ))

        return dependencies

    @staticmethod
    def _extract_typescript(code: str, language: Language) -> List[ExtractedDependency]:
        """Extract TypeScript imports (similar to JavaScript)."""
        return MultiLanguageExtractor._extract_javascript(code, language)

    @staticmethod
    def _extract_react(code: str, language: Language) -> List[ExtractedDependency]:
        """Extract React imports (similar to JavaScript)."""
        return MultiLanguageExtractor._extract_javascript(code, language)

    @staticmethod
    def _extract_java(code: str, language: Language) -> List[ExtractedDependency]:
        """Extract Java imports."""
        dependencies = []
        lines = code.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()

            # import package.Class;
            match = re.match(r'import\s+([a-zA-Z0-9_\.]+);', line)
            if match:
                full_import = match.group(1)
                # Extract package name (groupId:artifactId)
                parts = full_import.split('.')
                if len(parts) >= 2:
                    # Common Java package pattern: com.company.artifact
                    package_name = parts[-1]  # Use last part as artifact name
                    dependencies.append(ExtractedDependency(
                        name=package_name,
                        language=language,
                        import_statement=line,
                        line_number=i + 1
                    ))

        return dependencies

    @staticmethod
    def _extract_cpp(code: str, language: Language) -> List[ExtractedDependency]:
        """Extract C++ includes."""
        dependencies = []
        lines = code.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()

            # #include <library>
            match = re.match(r'#include\s+<([^>]+)>', line)
            if match:
                header = match.group(1)
                # Extract library name from header
                library = header.split('/')[0]
                if library not in ['iostream', 'vector', 'string', 'map']:  # Skip STL
                    dependencies.append(ExtractedDependency(
                        name=library,
                        language=language,
                        import_statement=line,
                        line_number=i + 1
                    ))

        return dependencies

    @staticmethod
    def _extract_go(code: str, language: Language) -> List[ExtractedDependency]:
        """Extract Go imports."""
        dependencies = []
        lines = code.split('\n')

        in_import_block = False

        for i, line in enumerate(lines):
            line = line.strip()

            # import "package"
            if line.startswith('import "'):
                match = re.match(r'import\s+"([^"]+)"', line)
                if match:
                    package = match.group(1)
                    dependencies.append(ExtractedDependency(
                        name=package,
                        language=language,
                        import_statement=line,
                        line_number=i + 1
                    ))

            # import ( ... )
            elif line.startswith('import ('):
                in_import_block = True
            elif in_import_block:
                if line == ')':
                    in_import_block = False
                else:
                    match = re.match(r'"([^"]+)"', line)
                    if match:
                        package = match.group(1)
                        dependencies.append(ExtractedDependency(
                            name=package,
                            language=language,
                            import_statement=line,
                            line_number=i + 1
                        ))

        return dependencies

    @staticmethod
    def _extract_rust(code: str, language: Language) -> List[ExtractedDependency]:
        """Extract Rust dependencies."""
        dependencies = []
        lines = code.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()

            # use crate_name::...;
            match = re.match(r'use\s+([a-zA-Z0-9_]+)::', line)
            if match:
                crate = match.group(1)
                if crate not in ['std', 'core', 'alloc']:  # Skip standard library
                    dependencies.append(ExtractedDependency(
                        name=crate,
                        language=language,
                        import_statement=line,
                        line_number=i + 1
                    ))

            # extern crate crate_name;
            match = re.match(r'extern\s+crate\s+([a-zA-Z0-9_]+);', line)
            if match:
                crate = match.group(1)
                dependencies.append(ExtractedDependency(
                    name=crate,
                    language=language,
                    import_statement=line,
                    line_number=i + 1
                ))

        return dependencies

    @staticmethod
    def _extract_csharp(code: str, language: Language) -> List[ExtractedDependency]:
        """Extract C# using directives."""
        dependencies = []
        lines = code.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()

            # using Package.Name;
            match = re.match(r'using\s+([a-zA-Z0-9_\.]+);', line)
            if match:
                namespace = match.group(1)
                # Extract package name (first part)
                package = namespace.split('.')[0]
                if package not in ['System', 'Microsoft']:  # Skip built-ins
                    dependencies.append(ExtractedDependency(
                        name=package,
                        language=language,
                        import_statement=line,
                        line_number=i + 1
                    ))

        return dependencies

    @staticmethod
    def get_unique_packages(dependencies: List[ExtractedDependency]) -> List[str]:
        """Get unique package names from dependencies."""
        return list(set(dep.name for dep in dependencies))


# Convenience function
def extract_packages_from_code(code: str, file_extension: str = None) -> Tuple[List[str], Language]:
    """
    Extract package names from code.

    Returns:
        Tuple of (package_names, detected_language)

    Examples:
        # Python
        packages, lang = extract_packages_from_code("import numpy as np")
        # Returns: (['numpy'], Language.PYTHON)

        # JavaScript
        packages, lang = extract_packages_from_code("import express from 'express'")
        # Returns: (['express'], Language.JAVASCRIPT)

        # React
        packages, lang = extract_packages_from_code("import React from 'react'", ".jsx")
        # Returns: (['react'], Language.REACT)
    """
    extractor = MultiLanguageExtractor()
    language = extractor.detect_language(code, file_extension)
    dependencies = extractor.extract_dependencies(code, language)
    packages = extractor.get_unique_packages(dependencies)

    return packages, language
