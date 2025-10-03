"""Tests for multi-language documentation finding."""

import pytest
import asyncio
from utils.multi_language_registry import (
    Language,
    RegistryFactory,
    NPMRegistry,
    MavenCentralRegistry,
    CratesIORegistry,
    GoPkgRegistry,
)
from utils.multi_language_doc_finder import find_multi_language_docs
from utils.multi_language_extractor import extract_packages_from_code, MultiLanguageExtractor


class TestLanguageDetection:
    """Test language detection from code and file extensions."""

    def test_detect_python(self):
        """Test Python detection."""
        code = "import numpy as np\nfrom pandas import DataFrame"
        lang = MultiLanguageExtractor.detect_language(code)
        assert lang == Language.PYTHON

    def test_detect_javascript(self):
        """Test JavaScript detection."""
        code = "const express = require('express');\nconst app = express();"
        lang = MultiLanguageExtractor.detect_language(code)
        assert lang == Language.JAVASCRIPT

    def test_detect_typescript(self):
        """Test TypeScript detection."""
        code = "interface User { name: string; age: number; }\nconst user: User = { name: 'John', age: 30 };"
        lang = MultiLanguageExtractor.detect_language(code)
        assert lang == Language.TYPESCRIPT

    def test_detect_react(self):
        """Test React detection."""
        code = "import React from 'react';\nimport { useState } from 'react';"
        lang = MultiLanguageExtractor.detect_language(code)
        assert lang == Language.REACT

    def test_detect_java(self):
        """Test Java detection."""
        code = "package com.example;\npublic class Main { public static void main(String[] args) {} }"
        lang = MultiLanguageExtractor.detect_language(code)
        assert lang == Language.JAVA

    def test_detect_rust(self):
        """Test Rust detection."""
        code = "fn main() { println!(\"Hello\"); }\nuse std::io;"
        lang = MultiLanguageExtractor.detect_language(code)
        assert lang == Language.RUST

    def test_detect_go(self):
        """Test Go detection."""
        code = "package main\nimport \"fmt\"\nfunc main() { fmt.Println(\"Hello\") }"
        lang = MultiLanguageExtractor.detect_language(code)
        assert lang == Language.GO

    def test_detect_from_extension(self):
        """Test detection from file extension."""
        assert MultiLanguageExtractor.detect_language("", ".py") == Language.PYTHON
        assert MultiLanguageExtractor.detect_language("", ".js") == Language.JAVASCRIPT
        assert MultiLanguageExtractor.detect_language("", ".jsx") == Language.REACT
        assert MultiLanguageExtractor.detect_language("", ".ts") == Language.TYPESCRIPT
        assert MultiLanguageExtractor.detect_language("", ".java") == Language.JAVA
        assert MultiLanguageExtractor.detect_language("", ".rs") == Language.RUST
        assert MultiLanguageExtractor.detect_language("", ".go") == Language.GO


class TestPackageExtraction:
    """Test package extraction from code."""

    def test_extract_python_imports(self):
        """Test Python import extraction."""
        code = """
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
"""
        packages, lang = extract_packages_from_code(code)
        assert lang == Language.PYTHON
        assert set(packages) == {'numpy', 'pandas', 'matplotlib', 'sklearn'}

    def test_extract_javascript_imports(self):
        """Test JavaScript import extraction."""
        code = """
import express from 'express';
import { Router } from 'express';
const axios = require('axios');
import React from 'react';
"""
        packages, lang = extract_packages_from_code(code)
        assert lang in [Language.JAVASCRIPT, Language.REACT]
        assert 'express' in packages
        assert 'axios' in packages
        assert 'react' in packages

    def test_extract_typescript_imports(self):
        """Test TypeScript import extraction."""
        code = """
import { FastifyInstance } from 'fastify';
import type { RouteOptions } from 'fastify';
import axios from 'axios';
"""
        packages, lang = extract_packages_from_code(code, ".ts")
        assert lang == Language.TYPESCRIPT
        assert 'fastify' in packages
        assert 'axios' in packages

    def test_extract_react_imports(self):
        """Test React import extraction."""
        code = """
import React, { useState, useEffect } from 'react';
import { BrowserRouter } from 'react-router-dom';
import styled from 'styled-components';
"""
        packages, lang = extract_packages_from_code(code, ".jsx")
        assert lang == Language.REACT
        assert set(packages) == {'react', 'react-router-dom', 'styled-components'}

    def test_extract_java_imports(self):
        """Test Java import extraction."""
        code = """
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import com.google.gson.Gson;
"""
        packages, lang = extract_packages_from_code(code)
        assert lang == Language.JAVA
        # Note: Java extraction returns class names, not Maven artifacts
        assert 'SpringApplication' in packages or 'Gson' in packages

    def test_extract_go_imports(self):
        """Test Go import extraction."""
        code = """
import (
    "fmt"
    "github.com/gin-gonic/gin"
    "github.com/joho/godotenv"
)
"""
        packages, lang = extract_packages_from_code(code)
        assert lang == Language.GO
        assert 'github.com/gin-gonic/gin' in packages
        assert 'github.com/joho/godotenv' in packages

    def test_extract_rust_imports(self):
        """Test Rust import extraction."""
        code = """
use tokio::runtime::Runtime;
use serde::{Serialize, Deserialize};
extern crate actix_web;
"""
        packages, lang = extract_packages_from_code(code)
        assert lang == Language.RUST
        assert 'tokio' in packages
        assert 'serde' in packages
        assert 'actix_web' in packages


@pytest.mark.asyncio
class TestPackageRegistries:
    """Test package registry integrations."""

    async def test_npm_registry(self):
        """Test NPM registry."""
        registry = NPMRegistry()
        try:
            info = await registry.get_package_info("express")
            assert info is not None
            assert info.name == "express"
            assert info.language == Language.JAVASCRIPT
            assert len(info.documentation_urls) > 0
        finally:
            await registry.close()

    async def test_npm_react_package(self):
        """Test NPM registry with React package."""
        registry = NPMRegistry()
        try:
            info = await registry.get_package_info("react")
            assert info is not None
            assert info.name == "react"
            assert len(info.documentation_urls) > 0
        finally:
            await registry.close()

    async def test_maven_central_registry(self):
        """Test Maven Central registry."""
        registry = MavenCentralRegistry()
        try:
            info = await registry.get_package_info("gson")
            assert info is not None
            assert info.language == Language.JAVA
            # Maven Central should return JavaDoc URLs
            assert any('javadoc.io' in url for url in info.documentation_urls)
        finally:
            await registry.close()

    async def test_crates_io_registry(self):
        """Test Crates.io registry."""
        registry = CratesIORegistry()
        try:
            info = await registry.get_package_info("tokio")
            assert info is not None
            assert info.name == "tokio"
            assert info.language == Language.RUST
            # Should include docs.rs
            assert any('docs.rs' in url for url in info.documentation_urls)
        finally:
            await registry.close()

    async def test_go_pkg_registry(self):
        """Test pkg.go.dev registry."""
        registry = GoPkgRegistry()
        try:
            info = await registry.get_package_info("github.com/gin-gonic/gin")
            assert info is not None
            assert info.language == Language.GO
            assert any('pkg.go.dev' in url for url in info.documentation_urls)
        finally:
            await registry.close()


@pytest.mark.asyncio
class TestDocumentationFinding:
    """Test multi-language documentation finding."""

    async def test_find_javascript_docs(self):
        """Test finding JavaScript documentation."""
        urls = await find_multi_language_docs(
            "express",
            language=Language.JAVASCRIPT
        )
        assert len(urls) > 0
        # Should include NPM or official docs
        assert any('express' in url.lower() for url in urls)

    async def test_find_react_docs(self):
        """Test finding React documentation."""
        urls = await find_multi_language_docs(
            "react",
            file_extension=".jsx"
        )
        assert len(urls) > 0

    async def test_find_typescript_docs(self):
        """Test finding TypeScript documentation."""
        urls = await find_multi_language_docs(
            "typescript",
            language=Language.TYPESCRIPT
        )
        assert len(urls) > 0

    async def test_find_java_docs(self):
        """Test finding Java documentation."""
        urls = await find_multi_language_docs(
            "gson",
            language=Language.JAVA
        )
        assert len(urls) > 0
        # Should include JavaDoc
        assert any('javadoc' in url.lower() for url in urls)

    async def test_find_rust_docs(self):
        """Test finding Rust documentation."""
        urls = await find_multi_language_docs(
            "tokio",
            language=Language.RUST
        )
        assert len(urls) > 0
        # Should include docs.rs
        assert any('docs.rs' in url for url in urls)

    async def test_find_go_docs(self):
        """Test finding Go documentation."""
        urls = await find_multi_language_docs(
            "github.com/gin-gonic/gin",
            language=Language.GO
        )
        assert len(urls) > 0
        # Should include pkg.go.dev
        assert any('pkg.go.dev' in url for url in urls)

    async def test_auto_detect_language(self):
        """Test automatic language detection."""
        code = "import React from 'react';"
        urls = await find_multi_language_docs(
            "react",
            code=code
        )
        assert len(urls) > 0


class TestRegistryFactory:
    """Test registry factory."""

    def test_get_javascript_registries(self):
        """Test getting JavaScript registries."""
        registries = RegistryFactory.get_registries(Language.JAVASCRIPT)
        assert len(registries) > 0
        assert any(isinstance(r, NPMRegistry) for r in registries)

    def test_get_java_registries(self):
        """Test getting Java registries."""
        registries = RegistryFactory.get_registries(Language.JAVA)
        assert len(registries) > 0
        assert any(isinstance(r, MavenCentralRegistry) for r in registries)

    def test_get_rust_registries(self):
        """Test getting Rust registries."""
        registries = RegistryFactory.get_registries(Language.RUST)
        assert len(registries) > 0
        assert any(isinstance(r, CratesIORegistry) for r in registries)

    @pytest.mark.asyncio
    async def test_factory_get_package_info(self):
        """Test factory get_package_info method."""
        info = await RegistryFactory.get_package_info(
            "express",
            language=Language.JAVASCRIPT
        )
        assert info is not None
        assert info.name == "express"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
