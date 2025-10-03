#!/usr/bin/env python3
"""
Multi-Language Documentation Finder - Demo Script

Demonstrates finding documentation for packages in various programming languages.
"""

import asyncio
from utils.multi_language_registry import Language
from utils.multi_language_doc_finder import find_multi_language_docs
from utils.multi_language_extractor import extract_packages_from_code


async def demo_javascript():
    """Demo JavaScript/TypeScript documentation finding."""
    print("\n" + "="*60)
    print("🟨 JAVASCRIPT/TYPESCRIPT DEMO")
    print("="*60)

    # JavaScript code example
    js_code = """
import express from 'express';
import axios from 'axios';
import { Router } from 'express';
const mongoose = require('mongoose');
"""

    print("\n📄 Code:")
    print(js_code)

    # Extract packages
    packages, lang = extract_packages_from_code(js_code)
    print(f"\n🔍 Detected Language: {lang.value}")
    print(f"📦 Extracted Packages: {packages}")

    # Find documentation for each package
    for package in packages:
        print(f"\n📚 Finding docs for: {package}")
        urls = await find_multi_language_docs(package, language=lang, max_urls=3)
        for url in urls:
            print(f"   ✅ {url}")


async def demo_react():
    """Demo React documentation finding."""
    print("\n" + "="*60)
    print("⚛️  REACT DEMO")
    print("="*60)

    react_code = """
import React, { useState, useEffect } from 'react';
import { BrowserRouter, Route } from 'react-router-dom';
import styled from 'styled-components';
import axios from 'axios';
"""

    print("\n📄 Code:")
    print(react_code)

    packages, lang = extract_packages_from_code(react_code, ".jsx")
    print(f"\n🔍 Detected Language: {lang.value}")
    print(f"📦 Extracted Packages: {packages}")

    for package in packages[:2]:  # Limit to 2 for demo
        print(f"\n📚 Finding docs for: {package}")
        urls = await find_multi_language_docs(package, language=lang, max_urls=3)
        for url in urls:
            print(f"   ✅ {url}")


async def demo_java():
    """Demo Java documentation finding."""
    print("\n" + "="*60)
    print("☕ JAVA DEMO")
    print("="*60)

    java_code = """
package com.example.app;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import com.google.gson.Gson;
import org.apache.commons.lang3.StringUtils;
"""

    print("\n📄 Code:")
    print(java_code)

    packages, lang = extract_packages_from_code(java_code)
    print(f"\n🔍 Detected Language: {lang.value}")
    print(f"📦 Extracted Packages: {packages}")

    # For Java, we'll search by common library names
    java_packages = ['gson', 'spring-boot', 'commons-lang3']

    for package in java_packages[:2]:
        print(f"\n📚 Finding docs for: {package}")
        urls = await find_multi_language_docs(package, language=Language.JAVA, max_urls=3)
        for url in urls:
            print(f"   ✅ {url}")


async def demo_rust():
    """Demo Rust documentation finding."""
    print("\n" + "="*60)
    print("🦀 RUST DEMO")
    print("="*60)

    rust_code = """
use tokio::runtime::Runtime;
use serde::{Serialize, Deserialize};
use actix_web::{web, App, HttpServer};
extern crate reqwest;
"""

    print("\n📄 Code:")
    print(rust_code)

    packages, lang = extract_packages_from_code(rust_code)
    print(f"\n🔍 Detected Language: {lang.value}")
    print(f"📦 Extracted Packages: {packages}")

    for package in packages[:2]:
        print(f"\n📚 Finding docs for: {package}")
        urls = await find_multi_language_docs(package, language=lang, max_urls=3)
        for url in urls:
            print(f"   ✅ {url}")


async def demo_go():
    """Demo Go documentation finding."""
    print("\n" + "="*60)
    print("🐹 GO DEMO")
    print("="*60)

    go_code = """
package main

import (
    "fmt"
    "github.com/gin-gonic/gin"
    "github.com/joho/godotenv"
)
"""

    print("\n📄 Code:")
    print(go_code)

    packages, lang = extract_packages_from_code(go_code)
    print(f"\n🔍 Detected Language: {lang.value}")
    print(f"📦 Extracted Packages: {packages}")

    for package in packages:
        if package.startswith("github.com"):
            print(f"\n📚 Finding docs for: {package}")
            urls = await find_multi_language_docs(package, language=lang, max_urls=3)
            for url in urls:
                print(f"   ✅ {url}")


async def demo_python():
    """Demo Python documentation finding."""
    print("\n" + "="*60)
    print("🐍 PYTHON DEMO")
    print("="*60)

    python_code = """
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from fastapi import FastAPI
"""

    print("\n📄 Code:")
    print(python_code)

    packages, lang = extract_packages_from_code(python_code)
    print(f"\n🔍 Detected Language: {lang.value}")
    print(f"📦 Extracted Packages: {packages}")

    for package in packages[:2]:
        print(f"\n📚 Finding docs for: {package}")
        urls = await find_multi_language_docs(package, language=lang, max_urls=3)
        for url in urls:
            print(f"   ✅ {url}")


async def demo_csharp():
    """Demo C# documentation finding."""
    print("\n" + "="*60)
    print("🔷 C# / .NET DEMO")
    print("="*60)

    csharp_code = """
using System;
using Newtonsoft.Json;
using Microsoft.AspNetCore.Mvc;
using Serilog;
"""

    print("\n📄 Code:")
    print(csharp_code)

    packages, lang = extract_packages_from_code(csharp_code)
    print(f"\n🔍 Detected Language: {lang.value}")
    print(f"📦 Extracted Packages: {packages}")

    # C# packages (NuGet names)
    csharp_packages = ['Newtonsoft.Json', 'Serilog']

    for package in csharp_packages:
        print(f"\n📚 Finding docs for: {package}")
        urls = await find_multi_language_docs(package, language=Language.CSHARP, max_urls=3)
        for url in urls:
            print(f"   ✅ {url}")


async def demo_multi_language_comparison():
    """Demo comparing documentation across languages."""
    print("\n" + "="*60)
    print("🌐 MULTI-LANGUAGE COMPARISON")
    print("="*60)

    test_cases = [
        ("express", Language.JAVASCRIPT, "Web framework"),
        ("fastapi", Language.PYTHON, "Web framework"),
        ("spring-boot", Language.JAVA, "Web framework"),
        ("actix-web", Language.RUST, "Web framework"),
        ("gin", Language.GO, "Web framework"),
    ]

    for package, lang, description in test_cases:
        print(f"\n📦 {package} ({lang.value}) - {description}")
        urls = await find_multi_language_docs(package, language=lang, max_urls=2)
        if urls:
            for url in urls:
                print(f"   ✅ {url}")
        else:
            print(f"   ❌ No documentation found")


async def main():
    """Run all demos."""
    print("\n" + "🌟"*30)
    print("  MULTI-LANGUAGE DOCUMENTATION FINDER - DEMO")
    print("🌟"*30)

    print("\nThis demo showcases finding documentation for packages in:")
    print("  • JavaScript/TypeScript")
    print("  • React")
    print("  • Python")
    print("  • Java")
    print("  • Rust")
    print("  • Go")
    print("  • C#/.NET")

    demos = [
        demo_javascript,
        demo_react,
        demo_python,
        demo_java,
        demo_rust,
        demo_go,
        demo_csharp,
        demo_multi_language_comparison,
    ]

    for demo in demos:
        try:
            await demo()
        except Exception as e:
            print(f"\n❌ Error in {demo.__name__}: {e}")

    print("\n" + "="*60)
    print("✨ DEMO COMPLETED")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("  ✅ Automatic language detection from code")
    print("  ✅ Package extraction from import statements")
    print("  ✅ Multi-registry support (NPM, Maven, Crates.io, etc.)")
    print("  ✅ Smart URL pattern matching")
    print("  ✅ Cross-language documentation finding")
    print()


if __name__ == "__main__":
    asyncio.run(main())
