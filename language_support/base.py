
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class ImportExtractor(ABC):
    """Abstract base class for extracting imports from code."""

    @abstractmethod
    def extract(self, code: str) -> List[str]:
        """Extract all imported libraries from code."""
        pass

class DocFinder(ABC):
    """Abstract base class for finding documentation URLs."""

    @abstractmethod
    async def find_documentation_urls(self, library_name: str, max_urls: int = 10) -> List[str]:
        """Find documentation URLs for a given library."""
        pass

class ErrorDetector(ABC):
    """Abstract base class for detecting errors in code."""

    @abstractmethod
    def detect_errors(self, code: str) -> List[Dict[str, Any]]:
        """Detect errors in the code and return a list of error details."""
        pass

class CodeFixer(ABC):
    """Abstract base class for fixing code."""

    @abstractmethod
    async def fix_code(self, code: str, errors: List[Dict[str, Any]], doc_context: str) -> str:
        """Fix the code based on the detected errors and documentation context."""
        pass

class LanguageSupport(ABC):
    """Abstract base class for language-specific support."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the language (e.g., 'python', 'javascript')."""
        pass

    @property
    @abstractmethod
    def import_extractor(self) -> ImportExtractor:
        """The import extractor for this language."""
        pass

    @property
    @abstractmethod
    def doc_finder(self) -> DocFinder:
        """The documentation finder for this language."""
        pass

    @property
    @abstractmethod
    def error_detector(self) -> ErrorDetector:
        """The error detector for this language."""
        pass

    @property
    @abstractmethod
    def code_fixer(self) -> CodeFixer:
        """The code fixer for this language."""
        pass
