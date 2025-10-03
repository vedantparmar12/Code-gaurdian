"""Multi-language universal documentation finder.

Extends the Python-only doc finder to support ALL major programming languages.
"""

import asyncio
import logging
from typing import List, Optional, Dict
from utils.multi_language_registry import (
    RegistryFactory,
    Language,
    PackageInfo
)
import httpx

logger = logging.getLogger(__name__)


class MultiLanguageDocFinder:
    """Find documentation for packages in ANY programming language."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

        # Language-specific documentation URL patterns
        self.doc_patterns = {
            Language.JAVASCRIPT: self._get_js_patterns,
            Language.TYPESCRIPT: self._get_ts_patterns,
            Language.REACT: self._get_react_patterns,
            Language.JAVA: self._get_java_patterns,
            Language.CPP: self._get_cpp_patterns,
            Language.GO: self._get_go_patterns,
            Language.RUST: self._get_rust_patterns,
            Language.CSHARP: self._get_csharp_patterns,
        }

    async def find_documentation_urls(
        self,
        package_name: str,
        language: Language = None,
        file_extension: str = None,
        code: str = None,
        max_urls: int = 10
    ) -> List[str]:
        """
        Find documentation URLs for ANY package in ANY language.

        Args:
            package_name: Name of the package/library
            language: Programming language (auto-detected if not provided)
            file_extension: File extension to help detect language
            code: Code snippet to help detect language
            max_urls: Maximum URLs to return

        Returns:
            List of valid documentation URLs
        """
        all_urls = []

        # Step 1: Try package registry (highest priority)
        registry_info = await RegistryFactory.get_package_info(
            package_name, language, file_extension, code
        )

        if registry_info:
            all_urls.extend(registry_info.documentation_urls)
            detected_language = registry_info.language
            logger.info(f"Found {len(registry_info.documentation_urls)} URLs from registry for {package_name}")
        else:
            # Fallback: try to detect language
            detected_language = RegistryFactory.detect_language(file_extension, code)
            if not detected_language:
                detected_language = language or Language.JAVASCRIPT  # Default fallback

        # Step 2: Try language-specific URL patterns
        pattern_generator = self.doc_patterns.get(detected_language)
        if pattern_generator:
            pattern_urls = await pattern_generator(package_name)
            all_urls.extend(pattern_urls)

        # Step 3: Try GitHub search
        github_urls = await self._search_github(package_name, detected_language)
        all_urls.extend(github_urls)

        # Step 4: Try web search as last resort
        web_urls = await self._web_search(package_name, detected_language)
        all_urls.extend(web_urls)

        # Validate and deduplicate URLs
        unique_urls = []
        seen = set()

        for url in all_urls:
            if url and url not in seen:
                if await self._validate_url(url):
                    unique_urls.append(url)
                    seen.add(url)

        logger.info(f"Found {len(unique_urls)} valid documentation URLs for {package_name}")
        return unique_urls[:max_urls]

    # Language-specific URL pattern generators

    async def _get_js_patterns(self, package_name: str) -> List[str]:
        """JavaScript/TypeScript documentation patterns."""
        patterns = [
            # NPM auto-generated docs
            f"https://www.npmjs.com/package/{package_name}",

            # Popular hosting platforms
            f"https://{package_name}.js.org/",
            f"https://{package_name}.dev/",

            # ReadTheDocs
            f"https://{package_name}.readthedocs.io/",

            # GitHub Pages
            f"https://{package_name}.github.io/",

            # Vercel/Netlify patterns
            f"https://{package_name}.vercel.app/",
            f"https://{package_name}.netlify.app/",
        ]

        return await self._probe_patterns(patterns)

    async def _get_ts_patterns(self, package_name: str) -> List[str]:
        """TypeScript-specific documentation patterns."""
        patterns = [
            # TypeDoc common patterns
            f"https://{package_name}.github.io/docs/",
            f"https://docs.{package_name}.dev/",

            # JSR registry
            f"https://jsr.io/@{package_name}",
            f"https://jsr.io/{package_name}",

            # Deno patterns
            f"https://deno.land/x/{package_name}",
        ]

        js_patterns = await self._get_js_patterns(package_name)
        return await self._probe_patterns(patterns) + js_patterns

    async def _get_react_patterns(self, package_name: str) -> List[str]:
        """React-specific documentation patterns."""
        patterns = [
            # Storybook
            f"https://{package_name}.netlify.app/",
            f"https://{package_name}-storybook.netlify.app/",

            # React-specific sites
            f"https://react-{package_name}.dev/",
            f"https://{package_name}-react.dev/",
        ]

        js_patterns = await self._get_js_patterns(package_name)
        return await self._probe_patterns(patterns) + js_patterns

    async def _get_java_patterns(self, package_name: str) -> List[str]:
        """Java documentation patterns."""
        patterns = [
            # JavaDoc
            f"https://javadoc.io/doc/{package_name}",

            # Spring framework
            f"https://docs.spring.io/spring-{package_name}/docs/current/reference/html/",

            # Apache projects
            f"https://{package_name}.apache.org/documentation.html",
        ]

        return await self._probe_patterns(patterns)

    async def _get_cpp_patterns(self, package_name: str) -> List[str]:
        """C++ documentation patterns."""
        patterns = [
            # Doxygen common patterns
            f"https://{package_name}.github.io/doxygen/",
            f"https://{package_name}.readthedocs.io/",

            # vcpkg
            f"https://vcpkg.io/en/packages/{package_name}.html",

            # Conan
            f"https://conan.io/center/{package_name}",
        ]

        return await self._probe_patterns(patterns)

    async def _get_go_patterns(self, package_name: str) -> List[str]:
        """Go documentation patterns."""
        patterns = [
            # pkg.go.dev
            f"https://pkg.go.dev/{package_name}",
            f"https://pkg.go.dev/github.com/{package_name}",

            # GoDoc (legacy)
            f"https://godoc.org/{package_name}",
        ]

        return await self._probe_patterns(patterns)

    async def _get_rust_patterns(self, package_name: str) -> List[str]:
        """Rust documentation patterns."""
        patterns = [
            # docs.rs (auto-generated)
            f"https://docs.rs/{package_name}",
            f"https://docs.rs/{package_name}/latest/{package_name}/",

            # crates.io
            f"https://crates.io/crates/{package_name}",
        ]

        return await self._probe_patterns(patterns)

    async def _get_csharp_patterns(self, package_name: str) -> List[str]:
        """.NET/C# documentation patterns."""
        patterns = [
            # Microsoft Docs
            f"https://docs.microsoft.com/en-us/dotnet/api/{package_name}",

            # NuGet
            f"https://www.nuget.org/packages/{package_name}",

            # .NET Foundation
            f"https://{package_name}.net/docs/",
        ]

        return await self._probe_patterns(patterns)

    async def _probe_patterns(self, patterns: List[str]) -> List[str]:
        """Probe URL patterns and return valid ones."""
        valid_urls = []

        tasks = [self._check_url_exists(url) for url in patterns]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, exists in zip(patterns, results):
            if exists is True:
                valid_urls.append(url)

        return valid_urls

    async def _check_url_exists(self, url: str) -> bool:
        """Check if a URL exists."""
        try:
            response = await self.client.head(url, timeout=5.0)
            return response.status_code in [200, 301, 302, 307, 308]
        except:
            return False

    async def _validate_url(self, url: str) -> bool:
        """Validate that a URL is accessible."""
        try:
            response = await self.client.head(url, timeout=10.0)
            return response.status_code in [200, 301, 302, 307, 308]
        except:
            return False

    async def _search_github(self, package_name: str, language: Language) -> List[str]:
        """Search GitHub for package documentation."""
        urls = []

        try:
            # Map language to GitHub language filter
            lang_map = {
                Language.JAVASCRIPT: "javascript",
                Language.TYPESCRIPT: "typescript",
                Language.REACT: "javascript",
                Language.JAVA: "java",
                Language.CPP: "c++",
                Language.GO: "go",
                Language.RUST: "rust",
                Language.CSHARP: "c#",
            }

            lang_filter = lang_map.get(language, "")
            search_url = "https://api.github.com/search/repositories"

            params = {
                "q": f"{package_name} language:{lang_filter}" if lang_filter else package_name,
                "sort": "stars",
                "order": "desc",
                "per_page": 3
            }

            response = await self.client.get(search_url, params=params)

            if response.status_code == 200:
                data = response.json()

                for repo in data.get('items', [])[:3]:
                    repo_name = repo.get('full_name')
                    homepage = repo.get('homepage')
                    default_branch = repo.get('default_branch', 'main')

                    if homepage:
                        urls.append(homepage)

                    urls.extend([
                        f"https://github.com/{repo_name}#readme",
                        f"https://github.com/{repo_name}/wiki",
                    ])

        except Exception as e:
            logger.warning(f"GitHub search error: {e}")

        return urls

    async def _web_search(self, package_name: str, language: Language) -> List[str]:
        """Web search as last resort using DuckDuckGo."""
        # This would integrate with existing web_search.py
        # For now, return empty list
        return []

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


# Convenience function
async def find_multi_language_docs(
    package_name: str,
    language: Language = None,
    file_extension: str = None,
    code: str = None,
    max_urls: int = 10
) -> List[str]:
    """
    Find documentation URLs for ANY package in ANY language.

    Examples:
        # JavaScript/TypeScript
        await find_multi_language_docs("react", file_extension=".jsx")
        await find_multi_language_docs("express", code="import express from 'express'")

        # Java
        await find_multi_language_docs("spring-boot", language=Language.JAVA)

        # Rust
        await find_multi_language_docs("tokio", file_extension=".rs")

        # Go
        await find_multi_language_docs("gin", language=Language.GO)
    """
    finder = MultiLanguageDocFinder()
    try:
        return await finder.find_documentation_urls(
            package_name, language, file_extension, code, max_urls
        )
    finally:
        await finder.close()
