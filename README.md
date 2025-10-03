# MCP Documentation Fetcher & Universal Multi-Language Code Validator

**🚀 Automatic code validation and fixing for 5+ languages using loop-based regeneration with Qwen3-Coder (480B) - Zero Claude tokens used!**

AI-powered MCP server that automatically validates and fixes code in **Python, JavaScript, TypeScript, Java, Go, and React** by fetching official documentation for any library. Works seamlessly with Claude Desktop, VS Code, Cursor, Windsurf, and any MCP-compatible IDE.

## 🎯 What It Does

**Problem:** Claude/AI writes code that has errors, uses wrong APIs, or incompatible versions. Fixing it costs Claude tokens and time.

**Solution:** This MCP server automatically (using **local Ollama**):
1. ✅ **Detects all errors** (syntax, imports, undefined names, type errors) in **5+ languages**
2. ✅ **Fetches official documentation** (PyPI for Python, npm for JS/TS, Maven for Java, pkg.go.dev for Go)
3. ✅ **Uses Qwen3-Coder 480B** in a loop to fix errors automatically
4. ✅ **Checks version compatibility** across all dependencies
5. ✅ **Validates library usage** against documentation
6. ✅ **Returns error-free code** - **NO CLAUDE TOKENS USED!** 🎉

## 🌐 Supported Languages

| Language | Import Detection | Doc Source | Error Detection | Status |
|----------|------------------|------------|-----------------|--------|
| **Python** | ✅ AST + Regex | PyPI API | ✅ AST + pylint | ✅ Production |
| **JavaScript** | ✅ Regex | npm Registry | ✅ Node.js --check | ✅ Production |
| **TypeScript** | ✅ Regex | npm Registry | ✅ tsc compiler | ✅ Production |
| **React/JSX** | ✅ Regex | React Docs + npm | ✅ tsc compiler | ✅ Production |
| **Java** | ✅ Regex | Maven Central | ✅ javac compiler | ✅ Production |
| **Go** | ✅ Regex | pkg.go.dev | ✅ go build | ✅ Production |

## ✨ Key Features

### Multi-Language Support
- 🌍 **6 Languages**: Python, JavaScript, TypeScript, React, Java, Go
- 🔌 **Pluggable Architecture**: Easy to add new languages
- 📦 **Universal Doc Fetcher**: Automatically finds docs from language-specific registries
- 🔍 **Language-Specific Error Detection**: Uses native compilers/linters

### Loop-Based Auto-Fixing
- 🔁 **Iterative Regeneration**: Automatically fixes code in iterations (up to 5 attempts)
- 🧠 **Qwen3-Coder 480B**: Uses powerful local LLM for intelligent code fixing
- 🚫 **Zero Claude Tokens**: All fixing happens locally via Ollama - saves money!
- 📚 **Documentation Context**: Fetches relevant docs to guide code fixing
- ⚡ **Fast Convergence**: Most errors fixed in 1-2 iterations

### Smart Features
- 🔍 **AST/Regex Parsing**: Extracts imports and dependencies accurately
- 🌐 **Multiple Search Strategies**: Primary registry → GitHub → Fallback search
- 🧠 **Semantic Search**: Ollama embeddings for intelligent documentation search
- 💾 **Persistent Caching**: SQLite-based cache with vector search
- 🔗 **MCP Compatible**: Works with Claude Desktop, VS Code, Cursor, Windsurf
- ⚡ **No Rate Limits**: Uses free APIs (PyPI, npm, Maven, pkg.go.dev, Ollama)

## 🚀 Quick Start

### Prerequisites

**Core Requirements:**
- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- Qwen3-Coder model: `ollama pull qwen3-coder:480b-cloud`
- Embedding model: `ollama pull nomic-embed-text`

**Language-Specific (Optional - for error detection):**
- **JavaScript/TypeScript**: Node.js 18+ (`node --version`)
- **TypeScript/React**: TypeScript compiler (`npm install -g typescript`)
- **Java**: JDK 11+ with javac (`javac -version`)
- **Go**: Go 1.19+ (`go version`)

*Note: Without these, the system falls back to regex-based error detection.*

### Installation

```bash
# Clone repository
git clone <repository-url>
cd mcp_doc_fetcher

# Install dependencies
pip install -r requirements.txt

# Pull Ollama models (REQUIRED for auto-fixing)
ollama pull qwen3-coder:480b-cloud  # Code generation & fixing
ollama pull nomic-embed-text         # Semantic search

# Configure environment
cp .env.example .env
# Edit .env with your Ollama settings
```

**Note:** Qwen3-Coder 480B is a large model (~480GB). Ensure you have sufficient disk space and RAM. For smaller setups, use `qwen3-coder:7b` instead.

### Configuration

Add to your Claude Desktop or MCP client configuration:

**Claude Desktop (macOS):** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Claude Desktop (Windows):** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "doc-fetcher": {
      "command": "python",
      "args": ["-m", "mcp_doc_fetcher.server"],
      "env": {
        "PYTHONPATH": "/path/to/mcp_doc_fetcher"
      }
    }
  }
}
```

## 💡 Usage

### With Claude Desktop, VS Code, Cursor, or Windsurf

Simply write code in **any supported language** and the MCP server will automatically validate and fix it:

#### Example 1: Python with FastAPI
```
You: Create a chatbot API with FastAPI and OpenAI

Claude: [writes code]

MCP: [automatically validates and fixes]
- Language: Python
- Extracts libraries: fastapi, openai, pydantic
- Fetches official documentation from PyPI
- Validates API usage
- Fixes deprecated methods
- Returns error-free code
```

#### Example 2: TypeScript/React
```
You: Create a Next.js API route with Prisma

Claude: [writes TypeScript code]

MCP: [automatically validates and fixes]
- Language: TypeScript
- Extracts libraries: next, prisma, react
- Fetches documentation from npm + Next.js docs
- Validates type errors with tsc
- Fixes import issues
- Returns error-free code
```

#### Example 3: Go with Gin
```
You: Create a REST API with Gin and GORM

Claude: [writes Go code]

MCP: [automatically validates and fixes]
- Language: Go
- Extracts libraries: github.com/gin-gonic/gin, gorm.io/gorm
- Fetches documentation from pkg.go.dev
- Validates with go build
- Fixes compilation errors
- Returns error-free code
```

### Available MCP Tool

**`validate_and_fix_code`** - Main tool that works for all languages
   ```json
   {
     "code": "your code here",
     "language": "python|javascript|typescript|java|go",
     "project_description": "optional context"
   }
   ```

**Supported language values:**
- `python` - Python code
- `javascript` - JavaScript/Node.js code
- `typescript` - TypeScript, TSX, React, Next.js code
- `java` - Java code
- `go` - Go/Golang code

## 🏗️ Architecture

```
┌──────────────────────────────────────────┐
│  USER: "Create chatbot with FastAPI"    │
└───────────────────┬──────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│  CLAUDE: Writes code                     │
└───────────────────┬──────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│  MCP: validate_and_fix_code()            │
│  1. Extract imports [fastapi, openai]    │
│  2. Fetch docs (PyPI API - universal!)   │
│  3. Validate against official docs       │
│  4. Fix errors automatically             │
└───────────────────┬──────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│  RESULT: Error-free, production code!    │
└──────────────────────────────────────────┘
```

### Core Components

1. **Language Support Modules** (`language_support/`)
   - **Base Classes**: Abstract interfaces for pluggable architecture
   - **Python**: AST parsing, PyPI API, Python error detection
   - **JavaScript**: Regex parsing, npm Registry, Node.js error detection
   - **TypeScript**: Regex parsing, npm + React/Next.js docs, tsc compiler
   - **Java**: Regex parsing, Maven Central, javac compiler
   - **Go**: Regex parsing, pkg.go.dev, go build

2. **Code Validator** (`utils/code_validator.py`)
   - Multi-language support orchestrator
   - Import extraction (language-specific)
   - Documentation fetching (registry-specific)
   - Error detection (compiler/linter-specific)
   - Automatic code fixing with Ollama

3. **Documentation Crawler** (`utils/crawler.py`)
   - Crawl4AI integration
   - Concurrent page fetching
   - Smart content extraction

4. **Semantic Search** (`utils/embeddings.py`)
   - Ollama embedding generation
   - Vector similarity search
   - Context-aware results

## 📋 Project Structure

```
mcp_doc_fetcher/
├── __init__.py                    # Package initialization
├── server.py                      # Main MCP server
├── config.py                      # Configuration
├── models.py                      # Pydantic models
├── requirements.txt               # Dependencies
├── language_support/              # 🆕 Multi-language support
│   ├── __init__.py
│   ├── base.py                    # Abstract base classes
│   ├── python.py                  # Python language support
│   ├── javascript.py              # JavaScript language support
│   ├── typescript.py              # TypeScript/React support
│   ├── java.py                    # Java language support
│   └── golang.py                  # Go language support
├── utils/
│   ├── code_validator.py          # Multi-language validator
│   ├── universal_doc_finder.py    # Legacy doc finder (Python)
│   ├── web_search.py              # Multi-strategy search
│   ├── crawler.py                 # Documentation crawling
│   ├── embeddings.py              # Semantic search
│   ├── cache.py                   # SQLite cache
│   ├── ollama_client.py           # Ollama integration
│   └── error_detector.py          # Python error detection
└── tests/
    ├── test_cache.py              # Cache tests
    └── test_language_support.py   # 🆕 Language module tests
```

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest tests/ -v

# Run language support tests
pytest tests/test_language_support.py -v

# Run with coverage
pytest tests/ -v --cov=mcp_doc_fetcher

# Run specific language tests
pytest tests/test_language_support.py::TestPythonImportExtractor -v
pytest tests/test_language_support.py::TestJavaScriptImportExtractor -v
pytest tests/test_language_support.py::TestTypeScriptImportExtractor -v
```

## 🔧 Configuration

Environment variables (`.env` file):

```bash
# Ollama Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Cache Configuration
CACHE_DB_PATH=./cache/embeddings.db
CACHE_MAX_AGE_HOURS=24

# Crawler Configuration
MAX_CONCURRENT_CRAWLS=5
CRAWL_TIMEOUT_SECONDS=30
MAX_PAGES_PER_LIBRARY=10
```

## 🎯 Examples

### Example 1: FastAPI + OpenAI Chatbot

**Input Code:**
```python
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()
client = OpenAI()

@app.post("/chat")
async def chat(message: str):
    response = client.completions.create(
        model="gpt-4",
        prompt=message
    )
    return response.choices[0].text
```

**MCP Validation:**
- ✅ Extracts: `fastapi`, `openai`
- ✅ Fetches docs from PyPI API
- ⚠️ Detects deprecated `completions` API
- ✅ Fixes to use `chat.completions`
- ✅ Adds proper Pydantic models

**Fixed Code:**
```python
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI()
client = OpenAI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": request.message}]
    )
    return {"response": response.choices[0].message.content}
```

## 🐛 Troubleshooting

### Server won't start
- Check Ollama is running: `curl http://localhost:11434/api/version`
- Verify Python version: `python --version` (requires 3.9+)
- Install dependencies: `pip install -r requirements.txt`

### No documentation found
- Library might not be on PyPI (try GitHub URL manually)
- Check library name spelling
- Try searching on PyPI: `https://pypi.org/search/?q=library_name`

### Crawling errors
- Some sites block automated crawling
- Try reducing `MAX_CONCURRENT_CRAWLS`
- Check internet connection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure tests pass (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) - LLM and embeddings
- [Crawl4AI](https://github.com/unclecode/crawl4ai) - Web crawling
- [Model Context Protocol](https://modelcontextprotocol.io) - MCP standard
- [PyPI](https://pypi.org) - Python package index API

## 📞 Support

- GitHub Issues: [Create an issue](https://github.com/your-username/mcp-doc-fetcher/issues)
- Documentation: See this README
- Questions: Open a discussion on GitHub

---

**Made with ❤️ for error-free AI code generation**
