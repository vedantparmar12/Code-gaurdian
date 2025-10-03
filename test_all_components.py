"""
Comprehensive test script for all MCP components.
Tests: Language support, AST extractors, Aho-Corasick, and MCP server.
"""

import sys
import asyncio
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*70)
    print("TEST 1: Module Imports")
    print("="*70)

    tests = []

    # Test language support
    try:
        from language_support.base import LanguageSupport, ImportExtractor
        tests.append(("language_support.base", True, None))
    except Exception as e:
        tests.append(("language_support.base", False, str(e)))

    try:
        from language_support.python import PythonLanguageSupport
        tests.append(("language_support.python", True, None))
    except Exception as e:
        tests.append(("language_support.python", False, str(e)))

    try:
        from language_support.javascript import JavaScriptLanguageSupport
        tests.append(("language_support.javascript", True, None))
    except Exception as e:
        tests.append(("language_support.javascript", False, str(e)))

    try:
        from language_support.typescript import TypeScriptLanguageSupport
        tests.append(("language_support.typescript", True, None))
    except Exception as e:
        tests.append(("language_support.typescript", False, str(e)))

    try:
        from language_support.java import JavaLanguageSupport
        tests.append(("language_support.java", True, None))
    except Exception as e:
        tests.append(("language_support.java", False, str(e)))

    try:
        from language_support.golang import GoLanguageSupport
        tests.append(("language_support.golang", True, None))
    except Exception as e:
        tests.append(("language_support.golang", False, str(e)))

    # Test utils
    try:
        from utils.aho_corasick_matcher import AhoCorasickMatcher
        tests.append(("utils.aho_corasick_matcher", True, None))
    except Exception as e:
        tests.append(("utils.aho_corasick_matcher", False, str(e)))

    try:
        from utils.ast_extractors import PythonASTExtractor
        tests.append(("utils.ast_extractors", True, None))
    except Exception as e:
        tests.append(("utils.ast_extractors", False, str(e)))

    try:
        from utils.code_validator import CodeValidator
        tests.append(("utils.code_validator", True, None))
    except Exception as e:
        tests.append(("utils.code_validator", False, str(e)))

    # Print results
    passed = sum(1 for _, success, _ in tests if success)
    total = len(tests)

    for module, success, error in tests:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {module}")
        if error:
            print(f"        Error: {error}")

    print(f"\nResult: {passed}/{total} imports successful")
    return passed == total


def test_python_import_extraction():
    """Test Python import extraction."""
    print("\n" + "="*70)
    print("TEST 2: Python Import Extraction")
    print("="*70)

    from language_support.python import PythonImportExtractor

    test_code = """
import requests
from fastapi import FastAPI
import openai
import os  # Standard library - should be filtered
import sys  # Standard library - should be filtered

def main():
    import json  # Standard library
    pass
"""

    extractor = PythonImportExtractor()
    packages = extractor.extract(test_code)

    print(f"Test Code:\n{test_code}")
    print(f"\nExtracted packages: {packages}")

    # Verify
    expected = {"requests", "fastapi", "openai"}
    actual = set(packages)

    success = expected == actual
    print(f"\nExpected: {sorted(expected)}")
    print(f"Actual:   {sorted(actual)}")
    print(f"Result: {'PASS' if success else 'FAIL'}")

    return success


def test_javascript_import_extraction():
    """Test JavaScript import extraction."""
    print("\n" + "="*70)
    print("TEST 3: JavaScript Import Extraction")
    print("="*70)

    from language_support.javascript import JavaScriptImportExtractor

    test_code = """
import express from 'express';
const axios = require('axios');
import { serve } from '@hono/node-server';
import fs from 'fs';  // Built-in - should be filtered
"""

    extractor = JavaScriptImportExtractor()
    packages = extractor.extract(test_code)

    print(f"Test Code:\n{test_code}")
    print(f"\nExtracted packages: {packages}")

    # Verify
    expected = {"express", "axios", "@hono/node-server"}
    actual = set(packages)

    success = expected == actual
    print(f"\nExpected: {sorted(expected)}")
    print(f"Actual:   {sorted(actual)}")
    print(f"Result: {'PASS' if success else 'FAIL'}")

    return success


def test_typescript_import_extraction():
    """Test TypeScript import extraction."""
    print("\n" + "="*70)
    print("TEST 4: TypeScript Import Extraction")
    print("="*70)

    from language_support.typescript import TypeScriptImportExtractor

    test_code = """
import React from 'react';
import { NextRequest } from 'next/server';
import type { User } from '@types/user';
"""

    extractor = TypeScriptImportExtractor()
    packages = extractor.extract(test_code)

    print(f"Test Code:\n{test_code}")
    print(f"\nExtracted packages: {packages}")

    # Verify
    expected = {"react", "next", "@types/user"}
    actual = set(packages)

    success = expected == actual
    print(f"\nExpected: {sorted(expected)}")
    print(f"Actual:   {sorted(actual)}")
    print(f"Result: {'PASS' if success else 'FAIL'}")

    return success


def test_java_import_extraction():
    """Test Java import extraction."""
    print("\n" + "="*70)
    print("TEST 5: Java Import Extraction")
    print("="*70)

    from language_support.java import JavaImportExtractor

    test_code = """
import org.springframework.boot.SpringApplication;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;  // Standard library - should be filtered
import javax.servlet.http.HttpServlet;  // Standard library - should be filtered
"""

    extractor = JavaImportExtractor()
    packages = extractor.extract(test_code)

    print(f"Test Code:\n{test_code}")
    print(f"\nExtracted packages: {packages}")

    # Verify (should not include java.* or javax.*)
    should_not_include = {"java", "javax"}
    actual = set(packages)

    success = not any(pkg.startswith("java.") or pkg.startswith("javax.") for pkg in packages)
    success = success and len(packages) > 0

    print(f"\nPackages: {sorted(packages)}")
    print(f"Correctly filtered java.* and javax.*: {success}")
    print(f"Result: {'PASS' if success else 'FAIL'}")

    return success


def test_go_import_extraction():
    """Test Go import extraction."""
    print("\n" + "="*70)
    print("TEST 6: Go Import Extraction")
    print("="*70)

    from language_support.golang import GoImportExtractor

    test_code = """
import (
    "fmt"  // Standard library - should be filtered
    "github.com/gin-gonic/gin"
    "gorm.io/gorm"
)
"""

    extractor = GoImportExtractor()
    packages = extractor.extract(test_code)

    print(f"Test Code:\n{test_code}")
    print(f"\nExtracted packages: {packages}")

    # Verify
    expected = {"github.com/gin-gonic/gin", "gorm.io/gorm"}
    actual = set(packages)

    success = expected == actual
    print(f"\nExpected: {sorted(expected)}")
    print(f"Actual:   {sorted(actual)}")
    print(f"Result: {'PASS' if success else 'FAIL'}")

    return success


def test_aho_corasick():
    """Test Aho-Corasick matcher."""
    print("\n" + "="*70)
    print("TEST 7: Aho-Corasick Pattern Matching")
    print("="*70)

    from utils.aho_corasick_matcher import AhoCorasickMatcher

    # Create matcher with patterns
    matcher = AhoCorasickMatcher()
    patterns = ["pickle.loads", "yaml.load", "eval(", "exec("]
    matcher.add_patterns(patterns)
    matcher.build()

    test_code = """
import pickle
import yaml

data = pickle.loads(user_input)  # DANGEROUS!
config = yaml.load(file_content)  # DANGEROUS!
result = eval(expression)         # DANGEROUS!
"""

    matches = matcher.search(test_code)

    print(f"Patterns: {patterns}")
    print(f"Test Code:\n{test_code}")
    print(f"\nMatches found: {len(matches)}")
    for pos, pattern in matches:
        print(f"  - '{pattern}' at position {pos}")

    # Verify
    found_patterns = set(pattern for _, pattern in matches)
    expected = {"pickle.loads", "yaml.load", "eval("}

    success = expected == found_patterns
    print(f"\nExpected to find: {sorted(expected)}")
    print(f"Actually found:   {sorted(found_patterns)}")
    print(f"Result: {'PASS' if success else 'FAIL'}")

    return success


def test_ast_extractor():
    """Test AST-based extraction."""
    print("\n" + "="*70)
    print("TEST 8: AST-Based Import Extraction")
    print("="*70)

    from utils.ast_extractors import PythonASTExtractor

    test_code = """
import requests
from fastapi import FastAPI
# import commented_import  # This should be ignored
message = "import fake_package"  # This should be ignored
"""

    result = PythonASTExtractor.extract_imports(test_code)

    print(f"Test Code:\n{test_code}")
    print(f"\nAST Extraction Result:")
    print(f"  Packages: {result['packages']}")
    print(f"  Detailed imports: {len(result['detailed'])} found")

    # Verify - should NOT include commented_import or fake_package
    packages = set(result['packages'])
    should_not_include = {"commented_import", "fake_package"}

    success = not any(pkg in should_not_include for pkg in packages)
    success = success and "requests" in packages and "fastapi" in packages

    print(f"\nCorrectly ignored comments/strings: {success}")
    print(f"Result: {'PASS' if success else 'FAIL'}")

    return success


def test_code_validator_registration():
    """Test CodeValidator language registration."""
    print("\n" + "="*70)
    print("TEST 9: CodeValidator Language Registration")
    print("="*70)

    from utils.code_validator import CodeValidator

    validator = CodeValidator()

    registered_languages = list(validator.languages.keys())

    print(f"Registered languages: {registered_languages}")

    expected_languages = {"python", "javascript", "typescript", "java", "go"}
    actual_languages = set(registered_languages)

    success = expected_languages == actual_languages

    print(f"\nExpected: {sorted(expected_languages)}")
    print(f"Actual:   {sorted(actual_languages)}")
    print(f"Result: {'PASS' if success else 'FAIL'}")

    return success


async def test_mcp_server():
    """Test MCP server initialization."""
    print("\n" + "="*70)
    print("TEST 10: MCP Server Initialization")
    print("="*70)

    try:
        from server import CodeFixerServer

        server = CodeFixerServer()

        print(f"Server name: {server.server.name}")
        print(f"Validator languages: {list(server.validator.languages.keys())}")

        success = server.server.name == "code-fixer"
        success = success and len(server.validator.languages) >= 5

        print(f"\nServer initialized: {success}")
        print(f"Result: {'PASS' if success else 'FAIL'}")

        return success
    except Exception as e:
        print(f"Error initializing server: {e}")
        print(f"Result: FAIL")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MCP Documentation Fetcher - Component Test Suite")
    print("="*70)

    results = []

    # Import tests
    results.append(("Module Imports", test_imports()))

    # Language-specific import extraction
    results.append(("Python Import Extraction", test_python_import_extraction()))
    results.append(("JavaScript Import Extraction", test_javascript_import_extraction()))
    results.append(("TypeScript Import Extraction", test_typescript_import_extraction()))
    results.append(("Java Import Extraction", test_java_import_extraction()))
    results.append(("Go Import Extraction", test_go_import_extraction()))

    # Advanced features
    results.append(("Aho-Corasick Matching", test_aho_corasick()))
    results.append(("AST Extraction", test_ast_extractor()))
    results.append(("CodeValidator Registration", test_code_validator_registration()))

    # MCP server
    results.append(("MCP Server Init", asyncio.run(test_mcp_server())))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {test_name}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests PASSED! MCP server is ready to use.")
        return 0
    else:
        print(f"\n{total - passed} tests FAILED. Please review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
