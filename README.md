# MCP Documentation Fetcher & Universal Code Validator

**🚀 Automatic code validation and fixing using loop-based regeneration with Qwen3-Coder (480B) - Zero Claude tokens used!**

AI-powered MCP server that automatically validates and fixes Python code by fetching official documentation for any library. Works seamlessly with Claude Desktop, VS Code, and any MCP-compatible IDE.

## 🎯 What It Does

**Problem:** Claude/AI writes code that has errors, uses wrong APIs, or incompatible versions. Fixing it costs Claude tokens and time.

**Solution:** This MCP server automatically (using **local Ollama**):
1. ✅ **Detects all errors** (syntax, imports, undefined names, type errors)
2. ✅ **Fetches official documentation** (works for ANY Python library!)
3. ✅ **Uses Qwen3-Coder 480B** in a loop to fix errors automatically
4. ✅ **Checks version compatibility** across all dependencies
5. ✅ **Validates library usage** against documentation
6. ✅ **Returns error-free code** - **NO CLAUDE TOKENS USED!** 🎉

## ✨ New Features (Loop-Based Auto-Fixing)

- 🔁 **Loop-Based Regeneration**: Automatically fixes code in iterations until error-free (up to 5 attempts)
- 🧠 **Qwen3-Coder 480B**: Uses powerful local LLM for intelligent code fixing
- 🚫 **Zero Claude Tokens**: All fixing happens locally via Ollama - saves money!
- 🔍 **Error Detection**: AST parsing, syntax checking, import validation, undefined name detection
- 📊 **Version Compatibility**: Checks PyPI for version conflicts and compatibility
- 📚 **Documentation Context**: Fetches relevant docs to guide code fixing
- 🔁 **Iterative Improvement**: Keeps fixing until code is error-free or max iterations reached

## ✨ Existing Features

- 🔍 **Universal Documentation Finder**: Uses PyPI API - works for ANY Python library (no hardcoding!)
- 🤖 **Smart Code Validation**: AST-based import extraction and pattern matching
- 🌐 **Multiple Search Strategies**: PyPI API → URL probing → GitHub → DuckDuckGo
- 🧠 **Semantic Search**: Ollama embeddings for intelligent documentation search
- 💾 **Persistent Caching**: SQLite-based cache with vector search
- 🔗 **MCP Compatible**: Works with Claude Desktop, VS Code, and other MCP clients
- ⚡ **No Rate Limits**: Uses free APIs (PyPI, DuckDuckGo, Ollama)

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- Qwen3-Coder model: `ollama pull qwen3-coder:480b-cloud`
- Embedding model: `ollama pull nomic-embed-text`

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

### With Claude Desktop

Simply write code and Claude will automatically validate and fix it:

```
You: Create a chatbot API with FastAPI and OpenAI

Claude: [writes code]

MCP: [automatically validates and fixes]
- Extracts libraries: fastapi, openai, pydantic
- Fetches official documentation
- Validates API usage
- Fixes deprecated methods
- Returns error-free code
```

### Available MCP Tools

1. **`fetch_documentation`**
   - Fetches and caches library documentation
   - Works for ANY Python library via PyPI API
   ```json
   {
     "library_name": "fastapi",
     "version": "latest",
     "max_pages": 10
   }
   ```

2. **`search_documentation`**
   - Semantic search within cached documentation
   ```json
   {
     "library_name": "fastapi",
     "query": "authentication examples",
     "max_results": 5
   }
   ```

3. **`validate_and_fix_code`** ⭐ Main tool
   - Validates and fixes code automatically
   ```json
   {
     "code": "your Python code here",
     "project_description": "chatbot API"
   }
   ```

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

1. **Universal Documentation Finder** (`utils/universal_doc_finder.py`)
   - PyPI API for official URLs (works for ALL packages!)
   - URL pattern probing for common doc locations
   - GitHub repository search
   - DuckDuckGo fallback search

2. **Code Validator** (`utils/code_validator.py`)
   - AST-based import extraction
   - Automatic documentation fetching
   - Pattern matching against official docs
   - Version compatibility checking
   - Automatic code fixing

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
├── utils/
│   ├── universal_doc_finder.py    # Find docs for ANY library
│   ├── code_validator.py          # Validate & fix code
│   ├── web_search.py              # Multi-strategy search
│   ├── crawler.py                 # Documentation crawling
│   ├── embeddings.py              # Semantic search
│   ├── cache.py                   # SQLite cache
│   └── doc_url_registry.py        # Fallback URL registry
└── tests/
    └── test_cache.py              # Unit tests
```

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=mcp_doc_fetcher

# Run specific test
pytest tests/test_cache.py -v
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
