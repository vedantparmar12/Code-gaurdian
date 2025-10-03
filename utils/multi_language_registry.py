"""Multi-language package registry system for universal documentation finding.

Supports:
- JavaScript/TypeScript: NPM, Yarn, JSR, Deno
- Java: Maven Central, Gradle
- C++: vcpkg, Conan
- Go: pkg.go.dev
- Rust: crates.io
- Ruby: RubyGems
- PHP: Packagist
- .NET: NuGet
- Python: PyPI (existing)
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import httpx

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    REACT = "react"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"
    CSHARP = "csharp"
    KOTLIN = "kotlin"
    SWIFT = "swift"


@dataclass
class PackageInfo:
    """Package information from registry."""
    name: str
    language: Language
    version: Optional[str] = None
    documentation_urls: List[str] = None
    homepage: Optional[str] = None
    repository: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        if self.documentation_urls is None:
            self.documentation_urls = []


class PackageRegistry:
    """Base class for package registry clients."""

    def __init__(self, language: Language):
        self.language = language
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def get_package_info(self, package_name: str) -> Optional[PackageInfo]:
        """Get package information from registry."""
        raise NotImplementedError

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class NPMRegistry(PackageRegistry):
    """NPM registry for JavaScript/TypeScript packages."""

    def __init__(self):
        super().__init__(Language.JAVASCRIPT)
        self.registry_url = "https://registry.npmjs.org"

    async def get_package_info(self, package_name: str) -> Optional[PackageInfo]:
        """Get package info from NPM."""
        try:
            response = await self.client.get(f"{self.registry_url}/{package_name}")

            if response.status_code == 200:
                data = response.json()
                latest_version = data.get('dist-tags', {}).get('latest', 'latest')
                latest_data = data.get('versions', {}).get(latest_version, data)

                # Extract documentation URLs
                doc_urls = []
                homepage = latest_data.get('homepage')
                repository = latest_data.get('repository', {})

                # Repository URL
                if isinstance(repository, dict):
                    repo_url = repository.get('url', '')
                else:
                    repo_url = repository or ''

                # Clean GitHub URLs
                if repo_url:
                    repo_url = repo_url.replace('git+', '').replace('.git', '')
                    repo_url = repo_url.replace('git://', 'https://')

                # Add homepage if it's a doc URL
                if homepage and self._is_doc_url(homepage):
                    doc_urls.append(homepage)

                # Add repository docs
                if repo_url and 'github.com' in repo_url:
                    doc_urls.append(f"{repo_url}#readme")
                    doc_urls.append(f"{repo_url}/blob/main/README.md")
                    doc_urls.append(f"{repo_url}/wiki")

                return PackageInfo(
                    name=package_name,
                    language=self.language,
                    version=latest_version,
                    documentation_urls=doc_urls,
                    homepage=homepage,
                    repository=repo_url,
                    description=latest_data.get('description')
                )

        except Exception as e:
            logger.error(f"NPM registry error for {package_name}: {e}")

        return None

    def _is_doc_url(self, url: str) -> bool:
        """Check if URL is likely documentation."""
        doc_indicators = ['docs', 'documentation', 'guide', 'wiki', 'api']
        return any(ind in url.lower() for ind in doc_indicators)


class JSRRegistry(PackageRegistry):
    """JSR (JavaScript Registry) for modern JavaScript/TypeScript."""

    def __init__(self):
        super().__init__(Language.JAVASCRIPT)
        self.api_url = "https://jsr.io/api"

    async def get_package_info(self, package_name: str) -> Optional[PackageInfo]:
        """Get package info from JSR."""
        try:
            # JSR uses @scope/name format
            response = await self.client.get(f"{self.api_url}/packages/{package_name}")

            if response.status_code == 200:
                data = response.json()

                doc_urls = [
                    f"https://jsr.io/{package_name}",  # JSR auto-generated docs
                    data.get('githubUrl'),
                ]
                doc_urls = [url for url in doc_urls if url]

                return PackageInfo(
                    name=package_name,
                    language=self.language,
                    documentation_urls=doc_urls,
                    homepage=data.get('homepage'),
                    repository=data.get('githubUrl'),
                    description=data.get('description')
                )

        except Exception as e:
            logger.error(f"JSR registry error for {package_name}: {e}")

        return None


class MavenCentralRegistry(PackageRegistry):
    """Maven Central for Java packages."""

    def __init__(self):
        super().__init__(Language.JAVA)
        self.search_url = "https://search.maven.org/solrsearch/select"

    async def get_package_info(self, package_name: str) -> Optional[PackageInfo]:
        """Get package info from Maven Central."""
        try:
            # Search for artifact
            params = {
                "q": f"a:{package_name}",
                "rows": 1,
                "wt": "json"
            }

            response = await self.client.get(self.search_url, params=params)

            if response.status_code == 200:
                data = response.json()
                docs = data.get('response', {}).get('docs', [])

                if docs:
                    doc = docs[0]
                    group_id = doc.get('g')
                    artifact_id = doc.get('a')
                    version = doc.get('latestVersion', doc.get('v'))

                    # Build documentation URLs
                    doc_urls = [
                        f"https://javadoc.io/doc/{group_id}/{artifact_id}/{version}",
                        f"https://www.javadoc.io/doc/{group_id}/{artifact_id}",
                        f"https://search.maven.org/artifact/{group_id}/{artifact_id}",
                    ]

                    return PackageInfo(
                        name=f"{group_id}:{artifact_id}",
                        language=self.language,
                        version=version,
                        documentation_urls=doc_urls,
                        description=f"Maven: {group_id}:{artifact_id}"
                    )

        except Exception as e:
            logger.error(f"Maven Central error for {package_name}: {e}")

        return None


class CratesIORegistry(PackageRegistry):
    """Crates.io for Rust packages."""

    def __init__(self):
        super().__init__(Language.RUST)
        self.api_url = "https://crates.io/api/v1"

    async def get_package_info(self, package_name: str) -> Optional[PackageInfo]:
        """Get package info from crates.io."""
        try:
            response = await self.client.get(
                f"{self.api_url}/crates/{package_name}",
                headers={"User-Agent": "mcp-doc-fetcher"}
            )

            if response.status_code == 200:
                data = response.json()
                crate = data.get('crate', {})

                doc_urls = [
                    f"https://docs.rs/{package_name}",  # docs.rs auto-generates docs
                    crate.get('documentation'),
                    crate.get('homepage'),
                    crate.get('repository'),
                ]
                doc_urls = [url for url in doc_urls if url]

                return PackageInfo(
                    name=package_name,
                    language=self.language,
                    version=crate.get('max_version'),
                    documentation_urls=doc_urls,
                    homepage=crate.get('homepage'),
                    repository=crate.get('repository'),
                    description=crate.get('description')
                )

        except Exception as e:
            logger.error(f"Crates.io error for {package_name}: {e}")

        return None


class GoPkgRegistry(PackageRegistry):
    """pkg.go.dev for Go packages."""

    def __init__(self):
        super().__init__(Language.GO)
        self.base_url = "https://pkg.go.dev"

    async def get_package_info(self, package_name: str) -> Optional[PackageInfo]:
        """Get package info from pkg.go.dev."""
        try:
            # Go packages are typically full import paths
            doc_url = f"{self.base_url}/{package_name}"

            # Check if page exists
            response = await self.client.head(doc_url)

            if response.status_code == 200:
                doc_urls = [
                    doc_url,
                    f"{doc_url}#section-documentation",
                ]

                # Try to get repository URL from page
                repo_url = None
                if 'github.com' in package_name:
                    repo_url = f"https://{package_name}"

                return PackageInfo(
                    name=package_name,
                    language=self.language,
                    documentation_urls=doc_urls,
                    repository=repo_url
                )

        except Exception as e:
            logger.error(f"Go pkg.dev error for {package_name}: {e}")

        return None


class NuGetRegistry(PackageRegistry):
    """.NET NuGet registry."""

    def __init__(self):
        super().__init__(Language.CSHARP)
        self.api_url = "https://api.nuget.org/v3"

    async def get_package_info(self, package_name: str) -> Optional[PackageInfo]:
        """Get package info from NuGet."""
        try:
            # NuGet API v3
            search_url = f"{self.api_url}/registration5-semver1/{package_name.lower()}/index.json"
            response = await self.client.get(search_url)

            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])

                if items:
                    latest = items[-1].get('items', [{}])[-1]
                    catalog_entry = latest.get('catalogEntry', {})

                    doc_urls = [
                        f"https://www.nuget.org/packages/{package_name}",
                        catalog_entry.get('projectUrl'),
                        catalog_entry.get('licenseUrl'),
                    ]
                    doc_urls = [url for url in doc_urls if url]

                    return PackageInfo(
                        name=package_name,
                        language=self.language,
                        version=catalog_entry.get('version'),
                        documentation_urls=doc_urls,
                        homepage=catalog_entry.get('projectUrl'),
                        description=catalog_entry.get('description')
                    )

        except Exception as e:
            logger.error(f"NuGet error for {package_name}: {e}")

        return None


class RegistryFactory:
    """Factory to get appropriate registry for a language/package."""

    # Map file extensions to languages
    EXTENSION_TO_LANGUAGE = {
        '.py': Language.PYTHON,
        '.js': Language.JAVASCRIPT,
        '.jsx': Language.REACT,
        '.ts': Language.TYPESCRIPT,
        '.tsx': Language.REACT,
        '.java': Language.JAVA,
        '.cpp': Language.CPP,
        '.cc': Language.CPP,
        '.cxx': Language.CPP,
        '.h': Language.CPP,
        '.hpp': Language.CPP,
        '.go': Language.GO,
        '.rs': Language.RUST,
        '.rb': Language.RUBY,
        '.php': Language.PHP,
        '.cs': Language.CSHARP,
        '.kt': Language.KOTLIN,
        '.swift': Language.SWIFT,
    }

    # Map languages to registries
    LANGUAGE_TO_REGISTRY = {
        Language.JAVASCRIPT: [NPMRegistry, JSRRegistry],
        Language.TYPESCRIPT: [NPMRegistry, JSRRegistry],
        Language.REACT: [NPMRegistry, JSRRegistry],
        Language.JAVA: [MavenCentralRegistry],
        Language.RUST: [CratesIORegistry],
        Language.GO: [GoPkgRegistry],
        Language.CSHARP: [NuGetRegistry],
    }

    @classmethod
    def detect_language(cls, file_extension: str = None, code: str = None) -> Optional[Language]:
        """Detect language from file extension or code content."""
        if file_extension:
            return cls.EXTENSION_TO_LANGUAGE.get(file_extension.lower())

        if code:
            # Simple heuristics for code detection
            if 'import ' in code and 'from ' in code:
                return Language.PYTHON
            elif 'import React' in code or 'from "react"' in code:
                return Language.REACT
            elif 'interface ' in code and ': ' in code:
                return Language.TYPESCRIPT
            elif 'public class' in code or 'package ' in code:
                return Language.JAVA
            elif 'fn main()' in code or 'use ' in code and '::' in code:
                return Language.RUST
            elif 'func ' in code and 'package ' in code:
                return Language.GO

        return None

    @classmethod
    def get_registries(cls, language: Language) -> List[PackageRegistry]:
        """Get all registries for a language."""
        registry_classes = cls.LANGUAGE_TO_REGISTRY.get(language, [])
        return [registry_class() for registry_class in registry_classes]

    @classmethod
    async def get_package_info(
        cls,
        package_name: str,
        language: Language = None,
        file_extension: str = None,
        code: str = None
    ) -> Optional[PackageInfo]:
        """Get package info, auto-detecting language if not provided."""

        # Detect language if not provided
        if language is None:
            language = cls.detect_language(file_extension, code)

        if language is None:
            logger.warning(f"Could not detect language for package: {package_name}")
            return None

        # Try all registries for the language
        registries = cls.get_registries(language)

        for registry in registries:
            try:
                info = await registry.get_package_info(package_name)
                if info:
                    return info
            finally:
                await registry.close()

        return None
