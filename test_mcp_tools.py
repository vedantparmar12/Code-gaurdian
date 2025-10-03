"""
Test MCP tools with real code samples to validate end-to-end functionality.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.code_validator import CodeValidator


async def test_python_validation():
    """Test Python code validation."""
    print("\n" + "="*70)
    print("MCP TOOL TEST 1: Python Code Validation")
    print("="*70)

    validator = CodeValidator()

    # Test code with intentional issues
    test_code = """
import requests
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root()
    return {"message": "Hello World"}
"""

    print(f"Test Code (has syntax error - missing colon):\n{test_code}")

    result = await validator.validate_and_fix_code(
        code=test_code,
        language="python",
        project_description="Simple FastAPI application",
        max_iterations=1
    )

    print(f"\nValidation Result:")
    print(f"  Original code: {len(result['original_code'])} chars")
    print(f"  Libraries found: {result.get('libraries_found', [])}")
    print(f"  Errors detected: {len(result.get('errors_detected', []))}")
    print(f"  Iterations: {result.get('iterations', 0)}")
    print(f"  Is error-free: {result.get('is_error_free', False)}")

    success = "libraries_found" in result and len(result["libraries_found"]) > 0
    print(f"\nResult: {'PASS' if success else 'FAIL'}")

    return success


async def test_javascript_validation():
    """Test JavaScript code validation."""
    print("\n" + "="*70)
    print("MCP TOOL TEST 2: JavaScript Code Validation")
    print("="*70)

    validator = CodeValidator()

    test_code = """
const express = require('express');
const axios = require('axios');

const app = express();

app.get('/', async (req, res) => {
    const data = await axios.get('https://api.example.com');
    res.json(data);
});

app.listen(3000);
"""

    print(f"Test Code:\n{test_code}")

    result = await validator.validate_and_fix_code(
        code=test_code,
        language="javascript",
        project_description="Express API",
        max_iterations=1
    )

    print(f"\nValidation Result:")
    print(f"  Libraries found: {result.get('libraries_found', [])}")
    print(f"  Is error-free: {result.get('is_error_free', False)}")

    expected_libs = {"express", "axios"}
    actual_libs = set(result.get("libraries_found", []))

    success = expected_libs.issubset(actual_libs)
    print(f"\nExpected libraries: {sorted(expected_libs)}")
    print(f"Found libraries: {sorted(actual_libs)}")
    print(f"Result: {'PASS' if success else 'FAIL'}")

    return success


async def test_typescript_validation():
    """Test TypeScript code validation."""
    print("\n" + "="*70)
    print("MCP TOOL TEST 3: TypeScript/React Code Validation")
    print("="*70)

    validator = CodeValidator()

    test_code = """
import React from 'react';
import { useState } from 'react';

const App = () => {
    const [count, setCount] = useState(0);

    return (
        <div>
            <h1>Count: {count}</h1>
            <button onClick={() => setCount(count + 1)}>Increment</button>
        </div>
    );
};

export default App;
"""

    print(f"Test Code:\n{test_code}")

    result = await validator.validate_and_fix_code(
        code=test_code,
        language="typescript",
        project_description="React counter component",
        max_iterations=1
    )

    print(f"\nValidation Result:")
    print(f"  Libraries found: {result.get('libraries_found', [])}")
    print(f"  Is error-free: {result.get('is_error_free', False)}")

    success = "react" in result.get("libraries_found", [])
    print(f"\nResult: {'PASS' if success else 'FAIL'}")

    return success


async def test_java_validation():
    """Test Java code validation."""
    print("\n" + "="*70)
    print("MCP TOOL TEST 4: Java Code Validation")
    print("="*70)

    validator = CodeValidator()

    test_code = """
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class DemoApplication {
    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
    }
}
"""

    print(f"Test Code:\n{test_code}")

    result = await validator.validate_and_fix_code(
        code=test_code,
        language="java",
        project_description="Spring Boot application",
        max_iterations=1
    )

    print(f"\nValidation Result:")
    print(f"  Libraries found: {result.get('libraries_found', [])}")
    print(f"  Is error-free: {result.get('is_error_free', False)}")

    libs = result.get("libraries_found", [])
    success = any("spring" in lib.lower() for lib in libs)

    print(f"\nResult: {'PASS' if success else 'FAIL'}")

    return success


async def test_go_validation():
    """Test Go code validation."""
    print("\n" + "="*70)
    print("MCP TOOL TEST 5: Go Code Validation")
    print("="*70)

    validator = CodeValidator()

    test_code = """
package main

import (
    "github.com/gin-gonic/gin"
    "net/http"
)

func main() {
    r := gin.Default()
    r.GET("/", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{"message": "Hello World"})
    })
    r.Run(":8080")
}
"""

    print(f"Test Code:\n{test_code}")

    result = await validator.validate_and_fix_code(
        code=test_code,
        language="go",
        project_description="Gin web server",
        max_iterations=1
    )

    print(f"\nValidation Result:")
    print(f"  Libraries found: {result.get('libraries_found', [])}")
    print(f"  Is error-free: {result.get('is_error_free', False)}")

    libs = result.get("libraries_found", [])
    success = any("gin" in lib.lower() for lib in libs)

    print(f"\nResult: {'PASS' if success else 'FAIL'}")

    return success


async def test_unsupported_language():
    """Test unsupported language handling."""
    print("\n" + "="*70)
    print("MCP TOOL TEST 6: Unsupported Language Handling")
    print("="*70)

    validator = CodeValidator()

    test_code = "println('Hello World')"

    result = await validator.validate_and_fix_code(
        code=test_code,
        language="ruby",  # Not supported
        project_description="Ruby script",
        max_iterations=1
    )

    print(f"Testing with unsupported language 'ruby'")
    print(f"\nValidation Result:")
    print(f"  Error: {result.get('error', 'No error')}")

    success = "error" in result and "not supported" in result["error"].lower()
    print(f"\nCorrectly rejected unsupported language: {success}")
    print(f"Result: {'PASS' if success else 'FAIL'}")

    return success


async def main():
    """Run all MCP tool tests."""
    print("\n" + "="*70)
    print("MCP Tools End-to-End Validation")
    print("="*70)

    results = []

    # Test each language
    results.append(("Python Validation", await test_python_validation()))
    results.append(("JavaScript Validation", await test_javascript_validation()))
    results.append(("TypeScript Validation", await test_typescript_validation()))
    results.append(("Java Validation", await test_java_validation()))
    results.append(("Go Validation", await test_go_validation()))
    results.append(("Unsupported Language", await test_unsupported_language()))

    # Summary
    print("\n" + "="*70)
    print("MCP TOOLS TEST SUMMARY")
    print("="*70)

    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {test_name}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\nTotal: {passed}/{total} MCP tool tests passed")

    if passed == total:
        print("\nAll MCP tools are working correctly!")
        print("\nSupported languages:")
        print("  - Python (PyPI documentation)")
        print("  - JavaScript (npm registry)")
        print("  - TypeScript/React (npm + official docs)")
        print("  - Java (Maven Central)")
        print("  - Go (pkg.go.dev)")
        return 0
    else:
        print(f"\n{total - passed} tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
