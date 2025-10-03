"""Language support modules for multi-language code validation and fixing."""

from .base import LanguageSupport, ImportExtractor, DocFinder, ErrorDetector, CodeFixer
from .python import PythonLanguageSupport
from .javascript import JavaScriptLanguageSupport
from .typescript import TypeScriptLanguageSupport
from .java import JavaLanguageSupport
from .golang import GoLanguageSupport

__all__ = [
    'LanguageSupport',
    'ImportExtractor',
    'DocFinder',
    'ErrorDetector',
    'CodeFixer',
    'PythonLanguageSupport',
    'JavaScriptLanguageSupport',
    'TypeScriptLanguageSupport',
    'JavaLanguageSupport',
    'GoLanguageSupport',
]
