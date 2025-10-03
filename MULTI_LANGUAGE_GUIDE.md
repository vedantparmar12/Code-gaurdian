# 🌐 Multi-Language Documentation Finder

**Universal documentation finding for ALL major programming languages!**

The MCP Documentation Fetcher now supports automatic documentation discovery for packages and libraries across **10+ programming languages**, using their respective package registries and smart URL pattern matching.

---

## 🎯 Supported Languages

| Language | Package Registry | Auto-Detection | Documentation Sources |
|----------|-----------------|----------------|----------------------|
| **Python** | PyPI | ✅ | PyPI, ReadTheDocs, GitHub |
| **JavaScript** | NPM, JSR | ✅ | NPM, ReadTheDocs, GitHub Pages |
| **TypeScript** | NPM, JSR | ✅ | TypeDoc, JSR, NPM |
| **React** | NPM | ✅ | NPM, Storybook, Official Docs |
| **Java** | Maven Central | ✅ | JavaDoc.io, Maven Central |
| **Rust** | Crates.io | ✅ | docs.rs, crates.io |
| **Go** | pkg.go.dev | ✅ | pkg.go.dev, GoDoc |
| **C++** | vcpkg, Conan | ✅ | Doxygen, GitHub |
| **C#/.NET** | NuGet | ✅ | NuGet, Microsoft Docs |
| **Kotlin** | Maven | ✅ | KDoc, Maven |
| **Swift** | CocoaPods | ✅ | Apple Docs, GitHub |

---

## 🚀 Quick Start

### 1. Basic Usage

```python
from utils.multi_language_doc_finder import find_multi_language_docs
from utils.multi_language_registry import Language

# JavaScript/TypeScript
urls = await find_multi_language_docs("express", language=Language.JAVASCRIPT)

# React
urls = await find_multi_language_docs("react-router-dom", file_extension=".jsx")

# Python
urls = await find_multi_language_docs("fastapi", language=Language.PYTHON)

# Java
urls = await find_multi_language_docs("gson", language=Language.JAVA)

# Rust
urls = await find_multi_language_docs("tokio", language=Language.RUST)

# Go
urls = await find_multi_language_docs("github.com/gin-gonic/gin", language=Language.GO)
```

### 2. Auto-Detection from Code

```python
from utils.multi_language_extractor import extract_packages_from_code

# Extract packages and detect language automatically
js_code = """
import express from 'express';
import axios from 'axios';
"""

packages, language = extract_packages_from_code(js_code)
print(packages)  # ['express', 'axios']
print(language)  # Language.JAVASCRIPT

# Then find docs
for package in packages:
    urls = await find_multi_language_docs(package, language=language)
    print(f"{package}: {urls}")
```

### 3. Run the Demo

```bash
# Run the multi-language demo
python multi_language_demo.py
```

---

## 📦 Package Registry Support

### JavaScript/TypeScript (NPM)

**Supported registries:**
- NPM (npm.js.org)
- JSR (jsr.io) - Modern JavaScript Registry
- Yarn Berry

**Example:**
```python
from utils.multi_language_registry import NPMRegistry

registry = NPMRegistry()
info = await registry.get_package_info("express")
print(info.documentation_urls)
# ['https://expressjs.com', 'https://github.com/expressjs/express#readme']
```

**Import detection:**
```javascript
// ES6 imports
import express from 'express';
import { Router } from 'express';

// CommonJS
const express = require('express');

// TypeScript
import type { Request } from 'express';
```

### Java (Maven Central)

**Supported registries:**
- Maven Central
- Gradle Plugin Portal

**Example:**
```python
from utils.multi_language_registry import MavenCentralRegistry

registry = MavenCentralRegistry()
info = await registry.get_package_info("gson")
print(info.documentation_urls)
# ['https://javadoc.io/doc/com.google.code.gson/gson/latest']
```

**Import detection:**
```java
import com.google.gson.Gson;
import org.springframework.boot.SpringApplication;
import org.apache.commons.lang3.StringUtils;
```

### Rust (Crates.io)

**Supported registries:**
- crates.io

**Example:**
```python
from utils.multi_language_registry import CratesIORegistry

registry = CratesIORegistry()
info = await registry.get_package_info("tokio")
print(info.documentation_urls)
# ['https://docs.rs/tokio', 'https://tokio.rs']
```

**Import detection:**
```rust
use tokio::runtime::Runtime;
use serde::{Serialize, Deserialize};
extern crate actix_web;
```

### Go (pkg.go.dev)

**Supported registries:**
- pkg.go.dev
- GoDoc (legacy)

**Example:**
```python
from utils.multi_language_registry import GoPkgRegistry

registry = GoPkgRegistry()
info = await registry.get_package_info("github.com/gin-gonic/gin")
print(info.documentation_urls)
# ['https://pkg.go.dev/github.com/gin-gonic/gin']
```

**Import detection:**
```go
import (
    "fmt"
    "github.com/gin-gonic/gin"
    "github.com/joho/godotenv"
)
```

### C#/.NET (NuGet)

**Supported registries:**
- NuGet.org

**Example:**
```python
from utils.multi_language_registry import NuGetRegistry

registry = NuGetRegistry()
info = await registry.get_package_info("Newtonsoft.Json")
print(info.documentation_urls)
# ['https://www.nuget.org/packages/Newtonsoft.Json', 'https://www.newtonsoft.com/json/help']
```

**Import detection:**
```csharp
using Newtonsoft.Json;
using Microsoft.AspNetCore.Mvc;
using Serilog;
```

---

## 🔍 Language Detection

The system automatically detects language from:

### 1. File Extension
```python
from utils.multi_language_registry import RegistryFactory

lang = RegistryFactory.detect_language(file_extension=".tsx")
# Returns: Language.REACT
```

### 2. Code Patterns
```python
code = "import React from 'react';"
lang = RegistryFactory.detect_language(code=code)
# Returns: Language.REACT
```

### Detection Heuristics

| Language | Detection Pattern |
|----------|------------------|
| Python | `import`, `from`, `def`, `__init__` |
| JavaScript | `const`, `let`, `require()`, `import` |
| TypeScript | `interface`, `: type`, `type` keyword |
| React | `import React`, `from 'react'`, JSX syntax |
| Java | `public class`, `package` |
| Rust | `fn main()`, `use ::`, `extern crate` |
| Go | `func`, `package`, `import (...)` |
| C# | `using`, `namespace`, `class` |

---

## 📚 Documentation Finding Strategy

The system uses a **multi-strategy approach** to find documentation:

### Strategy Flow

```
1. Package Registry (Highest Priority)
   ├─ NPM for JavaScript/TypeScript
   ├─ Maven Central for Java
   ├─ Crates.io for Rust
   ├─ pkg.go.dev for Go
   └─ NuGet for C#

2. Language-Specific URL Patterns
   ├─ ReadTheDocs patterns
   ├─ Official doc domains
   └─ GitHub Pages patterns

3. GitHub Repository Search
   ├─ Search by package name
   ├─ Filter by language
   └─ Extract README/Wiki URLs

4. Web Search (DuckDuckGo)
   └─ Last resort fallback
```

### Example URL Patterns

**JavaScript/TypeScript:**
- `https://npmjs.com/package/{name}`
- `https://{name}.js.org/`
- `https://jsr.io/{name}`
- `https://{name}.dev/`

**Java:**
- `https://javadoc.io/doc/{groupId}/{artifactId}`
- `https://docs.spring.io/spring-{name}/`
- `https://{name}.apache.org/documentation.html`

**Rust:**
- `https://docs.rs/{name}`
- `https://crates.io/crates/{name}`

**Go:**
- `https://pkg.go.dev/{import-path}`
- `https://godoc.org/{import-path}`

---

## 🧪 Testing

### Run Tests

```bash
# Run all multi-language tests
pytest tests/test_multi_language.py -v

# Run specific test
pytest tests/test_multi_language.py::TestLanguageDetection -v

# Run with coverage
pytest tests/test_multi_language.py --cov=utils
```

### Test Coverage

- ✅ Language detection (file extension & code patterns)
- ✅ Package extraction for all languages
- ✅ Registry integration (NPM, Maven, Crates.io, etc.)
- ✅ Documentation URL finding
- ✅ Auto-detection workflow

---

## 🔧 Configuration

### Environment Variables

```bash
# Multi-language support (add to .env)

# NPM Registry
NPM_REGISTRY_URL=https://registry.npmjs.org

# Maven Central
MAVEN_SEARCH_URL=https://search.maven.org/solrsearch/select

# Crates.io
CRATES_IO_API=https://crates.io/api/v1

# pkg.go.dev
GO_PKG_URL=https://pkg.go.dev

# NuGet
NUGET_API_URL=https://api.nuget.org/v3
```

### Custom Patterns

You can add custom documentation URL patterns in `multi_language_doc_finder.py`:

```python
# Add custom patterns for a language
async def _get_custom_patterns(self, package_name: str) -> List[str]:
    patterns = [
        f"https://custom-domain.com/{package_name}/docs/",
        f"https://docs.{package_name}.custom/",
    ]
    return await self._probe_patterns(patterns)
```

---

## 💡 Use Cases

### 1. AI Code Validation (Multi-Language)

```python
from utils.multi_language_extractor import extract_packages_from_code
from utils.multi_language_doc_finder import find_multi_language_docs

# User's code in any language
user_code = """
import express from 'express';
import axios from 'axios';

const app = express();
app.get('/', async (req, res) => {
    const data = await axios.get('https://api.example.com');
    res.json(data.data);
});
"""

# Extract packages and detect language
packages, language = extract_packages_from_code(user_code)

# Find documentation for validation
for package in packages:
    docs = await find_multi_language_docs(package, language=language)
    print(f"Validating {package} against: {docs}")
```

### 2. Cross-Language Project Analysis

```python
# Analyze a polyglot project
project_files = {
    "backend.py": "import fastapi\nfrom sqlalchemy import create_engine",
    "frontend.tsx": "import React from 'react'\nimport axios from 'axios'",
    "mobile.swift": "import Alamofire\nimport SwiftUI"
}

for file, code in project_files.items():
    packages, lang = extract_packages_from_code(code, file_extension=f".{file.split('.')[-1]}")
    print(f"\n{file} ({lang.value}):")
    for pkg in packages:
        urls = await find_multi_language_docs(pkg, language=lang, max_urls=2)
        print(f"  {pkg}: {urls}")
```

### 3. Documentation Aggregation

```python
# Aggregate docs for a tech stack
tech_stack = {
    "frontend": [("react", Language.REACT), ("tailwindcss", Language.JAVASCRIPT)],
    "backend": [("fastapi", Language.PYTHON), ("sqlalchemy", Language.PYTHON)],
    "mobile": [("flutter", Language.DART)],
}

docs_map = {}
for category, packages in tech_stack.items():
    docs_map[category] = {}
    for package, lang in packages:
        urls = await find_multi_language_docs(package, language=lang)
        docs_map[category][package] = urls

print(docs_map)
```

---

## 🛠️ API Reference

### Core Functions

#### `find_multi_language_docs()`
```python
async def find_multi_language_docs(
    package_name: str,
    language: Language = None,
    file_extension: str = None,
    code: str = None,
    max_urls: int = 10
) -> List[str]:
    """
    Find documentation URLs for any package in any language.

    Args:
        package_name: Package/library name
        language: Programming language (auto-detected if None)
        file_extension: File extension to help detect language
        code: Code snippet to help detect language
        max_urls: Maximum URLs to return

    Returns:
        List of valid documentation URLs
    """
```

#### `extract_packages_from_code()`
```python
def extract_packages_from_code(
    code: str,
    file_extension: str = None
) -> Tuple[List[str], Language]:
    """
    Extract package names from code.

    Args:
        code: Source code
        file_extension: File extension (.py, .js, .java, etc.)

    Returns:
        Tuple of (package_names, detected_language)
    """
```

### Registry Classes

All registries implement the `PackageRegistry` interface:

```python
class PackageRegistry:
    async def get_package_info(self, package_name: str) -> Optional[PackageInfo]
    async def close(self)
```

**Available Registries:**
- `NPMRegistry` - JavaScript/TypeScript
- `JSRRegistry` - Modern JavaScript Registry
- `MavenCentralRegistry` - Java
- `CratesIORegistry` - Rust
- `GoPkgRegistry` - Go
- `NuGetRegistry` - C#/.NET

---

## 🎓 Examples

### JavaScript/TypeScript Example

```python
# Find Express.js documentation
urls = await find_multi_language_docs("express", language=Language.JAVASCRIPT)
print(urls)
# ['https://expressjs.com', 'https://github.com/expressjs/express#readme']

# Auto-detect from code
code = "import express from 'express';"
packages, lang = extract_packages_from_code(code)
# packages: ['express'], lang: Language.JAVASCRIPT
```

### React Example

```python
# Find React documentation
urls = await find_multi_language_docs("react", file_extension=".jsx")
print(urls)
# ['https://react.dev', 'https://www.npmjs.com/package/react']

# Extract from React code
code = """
import React, { useState } from 'react';
import { BrowserRouter } from 'react-router-dom';
"""
packages, lang = extract_packages_from_code(code, ".jsx")
# packages: ['react', 'react-router-dom'], lang: Language.REACT
```

### Java Example

```python
# Find Gson documentation
urls = await find_multi_language_docs("gson", language=Language.JAVA)
print(urls)
# ['https://javadoc.io/doc/com.google.code.gson/gson']

# Extract from Java code
code = "import com.google.gson.Gson;"
packages, lang = extract_packages_from_code(code)
# packages: ['Gson'], lang: Language.JAVA
```

---

## 🔒 Rate Limits & Best Practices

### Registry Rate Limits

| Registry | Rate Limit | Notes |
|----------|-----------|-------|
| NPM | No official limit | Be reasonable |
| Maven Central | No official limit | Public API |
| Crates.io | 1 req/sec | Requires User-Agent |
| pkg.go.dev | No limit | Public service |
| NuGet | No official limit | Public API |

### Best Practices

1. **Cache results** - Use the built-in caching system
2. **Batch requests** - Process multiple packages together
3. **Handle errors gracefully** - Some registries may be unavailable
4. **Respect rate limits** - Add delays if needed

```python
# Example: Batch processing with caching
from utils.cache import DocumentationCache

cache = DocumentationCache()

async def get_docs_with_cache(package: str, language: Language):
    # Check cache first
    cached = await cache.get(package, language)
    if cached:
        return cached

    # Fetch and cache
    urls = await find_multi_language_docs(package, language=language)
    await cache.set(package, language, urls)
    return urls
```

---

## 🐛 Troubleshooting

### Issue: Language not detected

**Solution:** Specify language explicitly
```python
urls = await find_multi_language_docs("package-name", language=Language.JAVASCRIPT)
```

### Issue: No documentation found

**Possible causes:**
1. Package name misspelled
2. Package not in registry
3. Documentation not publicly available

**Solution:** Try alternative names or check registry directly
```python
# Try with registry-specific name format
# Java: groupId:artifactId format
urls = await find_multi_language_docs("com.google.code.gson:gson", language=Language.JAVA)

# Go: Full import path
urls = await find_multi_language_docs("github.com/gin-gonic/gin", language=Language.GO)
```

### Issue: Wrong packages extracted

**Solution:** Provide file extension for better detection
```python
packages, lang = extract_packages_from_code(code, file_extension=".tsx")
```

---

## 📈 Future Enhancements

- [ ] Add more language support (Dart, Kotlin, Swift)
- [ ] Support for monorepo package resolution
- [ ] Version-specific documentation links
- [ ] IDE integration (VS Code extension)
- [ ] Documentation quality scoring
- [ ] Offline documentation caching

---

## 🤝 Contributing

To add support for a new language:

1. **Create registry class** in `multi_language_registry.py`
2. **Add detection logic** in `RegistryFactory`
3. **Add extraction patterns** in `multi_language_extractor.py`
4. **Add URL patterns** in `multi_language_doc_finder.py`
5. **Write tests** in `tests/test_multi_language.py`

See existing implementations for examples!

---

## 📝 License

MIT License - Same as the main project

---

**Happy coding across all languages! 🚀**
