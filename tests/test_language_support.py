"""Unit tests for language support modules."""

import pytest
from language_support.python import PythonImportExtractor
from language_support.javascript import JavaScriptImportExtractor
from language_support.typescript import TypeScriptImportExtractor
from language_support.java import JavaImportExtractor
from language_support.golang import GoImportExtractor


class TestPythonImportExtractor:
    """Test Python import extraction."""

    def test_simple_import(self):
        code = "import requests"
        extractor = PythonImportExtractor()
        libs = extractor.extract(code)
        assert "requests" in libs

    def test_from_import(self):
        code = "from fastapi import FastAPI"
        extractor = PythonImportExtractor()
        libs = extractor.extract(code)
        assert "fastapi" in libs

    def test_multiple_imports(self):
        code = """
import requests
from fastapi import FastAPI
import openai
"""
        extractor = PythonImportExtractor()
        libs = extractor.extract(code)
        assert "requests" in libs
        assert "fastapi" in libs
        assert "openai" in libs

    def test_exclude_standard_library(self):
        code = "import os\nimport sys\nimport requests"
        extractor = PythonImportExtractor()
        libs = extractor.extract(code)
        assert "os" not in libs
        assert "sys" not in libs
        assert "requests" in libs

    def test_syntax_error_fallback(self):
        code = "import requests\nimport broken syntax here\nfrom fastapi import FastAPI"
        extractor = PythonImportExtractor()
        libs = extractor.extract(code)
        # Should still extract using regex
        assert len(libs) > 0


class TestJavaScriptImportExtractor:
    """Test JavaScript import extraction."""

    def test_es6_import(self):
        code = "import express from 'express';"
        extractor = JavaScriptImportExtractor()
        libs = extractor.extract(code)
        assert "express" in libs

    def test_require(self):
        code = "const axios = require('axios');"
        extractor = JavaScriptImportExtractor()
        libs = extractor.extract(code)
        assert "axios" in libs

    def test_scoped_package(self):
        code = "import { serve } from '@hono/node-server';"
        extractor = JavaScriptImportExtractor()
        libs = extractor.extract(code)
        assert "@hono/node-server" in libs

    def test_exclude_builtins(self):
        code = "import fs from 'fs';\nimport express from 'express';"
        extractor = JavaScriptImportExtractor()
        libs = extractor.extract(code)
        assert "fs" not in libs
        assert "express" in libs

    def test_exclude_relative_imports(self):
        code = "import foo from './foo';\nimport express from 'express';"
        extractor = JavaScriptImportExtractor()
        libs = extractor.extract(code)
        assert "express" in libs
        assert len(libs) == 1


class TestTypeScriptImportExtractor:
    """Test TypeScript import extraction."""

    def test_typescript_import(self):
        code = "import { NextRequest } from 'next/server';"
        extractor = TypeScriptImportExtractor()
        libs = extractor.extract(code)
        assert "next" in libs

    def test_react_import(self):
        code = "import React from 'react';\nimport { useState } from 'react';"
        extractor = TypeScriptImportExtractor()
        libs = extractor.extract(code)
        assert "react" in libs

    def test_type_imports(self):
        code = "import type { User } from '@types/user';"
        extractor = TypeScriptImportExtractor()
        libs = extractor.extract(code)
        assert "@types/user" in libs

    def test_dynamic_import(self):
        code = "const module = await import('lodash');"
        extractor = TypeScriptImportExtractor()
        libs = extractor.extract(code)
        assert "lodash" in libs


class TestJavaImportExtractor:
    """Test Java import extraction."""

    def test_simple_import(self):
        code = "import org.springframework.boot.SpringApplication;"
        extractor = JavaImportExtractor()
        libs = extractor.extract(code)
        assert "org.springframework.boot" in libs

    def test_wildcard_import(self):
        code = "import com.google.common.collect.*;"
        extractor = JavaImportExtractor()
        libs = extractor.extract(code)
        assert "com.google.common" in libs

    def test_exclude_java_standard(self):
        code = """
import java.util.List;
import javax.servlet.http.HttpServlet;
import org.springframework.boot.SpringApplication;
"""
        extractor = JavaImportExtractor()
        libs = extractor.extract(code)
        assert "org.springframework.boot" in libs
        assert not any("java." in lib for lib in libs)
        assert not any("javax." in lib for lib in libs)

    def test_multiple_packages(self):
        code = """
import org.springframework.boot.SpringApplication;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.hibernate.Session;
"""
        extractor = JavaImportExtractor()
        libs = extractor.extract(code)
        assert len(libs) >= 3


class TestGoImportExtractor:
    """Test Go import extraction."""

    def test_simple_import(self):
        code = 'import "github.com/gin-gonic/gin"'
        extractor = GoImportExtractor()
        libs = extractor.extract(code)
        assert "github.com/gin-gonic/gin" in libs

    def test_grouped_imports(self):
        code = '''
import (
    "github.com/gin-gonic/gin"
    "github.com/lib/pq"
    "gorm.io/gorm"
)
'''
        extractor = GoImportExtractor()
        libs = extractor.extract(code)
        assert "github.com/gin-gonic/gin" in libs
        assert "github.com/lib/pq" in libs
        assert "gorm.io/gorm" in libs

    def test_exclude_standard_library(self):
        code = '''
import (
    "fmt"
    "net/http"
    "github.com/gin-gonic/gin"
)
'''
        extractor = GoImportExtractor()
        libs = extractor.extract(code)
        assert "fmt" not in libs
        assert "net/http" not in libs
        assert "github.com/gin-gonic/gin" in libs

    def test_aliased_import(self):
        code = 'import mux "github.com/gorilla/mux"'
        extractor = GoImportExtractor()
        libs = extractor.extract(code)
        assert "github.com/gorilla/mux" in libs


@pytest.mark.asyncio
class TestDocFinders:
    """Test documentation finders for all languages."""

    async def test_python_doc_finder(self):
        from language_support.python import PythonDocFinder
        finder = PythonDocFinder()
        urls = await finder.find_documentation_urls("requests")
        assert len(urls) > 0
        assert any("requests" in url.lower() for url in urls)

    async def test_javascript_doc_finder(self):
        from language_support.javascript import JavaScriptDocFinder
        finder = JavaScriptDocFinder()
        urls = await finder.find_documentation_urls("express")
        assert len(urls) > 0

    async def test_typescript_doc_finder_react(self):
        from language_support.typescript import TypeScriptDocFinder
        finder = TypeScriptDocFinder()
        urls = await finder.find_documentation_urls("react")
        assert len(urls) > 0
        assert any("react.dev" in url or "reactjs.org" in url for url in urls)

    async def test_typescript_doc_finder_nextjs(self):
        from language_support.typescript import TypeScriptDocFinder
        finder = TypeScriptDocFinder()
        urls = await finder.find_documentation_urls("next")
        assert len(urls) > 0
        assert any("nextjs.org" in url for url in urls)

    async def test_go_doc_finder(self):
        from language_support.golang import GoDocFinder
        finder = GoDocFinder()
        urls = await finder.find_documentation_urls("github.com/gin-gonic/gin")
        assert len(urls) > 0
        assert any("pkg.go.dev" in url for url in urls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
