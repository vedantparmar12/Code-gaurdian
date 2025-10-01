# MCP Documentation Fetcher - Integration Guide

## Universal AI Code Validator with Loop-Based Auto-Fixing

This MCP server provides **automatic code validation and fixing** using official documentation and the Qwen3-Coder model. It works with Claude Desktop, VS Code, and any MCP-compatible IDE.

---

## 🎯 What This Does

When Claude (or any AI) generates code, this MCP server:

1. **Detects all errors** (syntax, imports, undefined names, etc.)
2. **Fetches official documentation** for all libraries used
3. **Uses Qwen3-Coder (480B)** to fix errors in a loop
4. **Returns error-free code** without costing Claude tokens
5. **Checks version compatibility** across all dependencies

### Visual Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ User: "Create a FastAPI chatbot"                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Claude Code generates broken code:                           │
│   import fastapi                                             │
│   app = FastAPI(  # <-- Missing )                           │
│   @app.get("/chat")                                          │
│   def chat(msg):                                             │
│       return process(msg)  # <-- Undefined function         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ MCP Server Auto-Fixes (NO CLAUDE TOKENS USED)               │
│                                                               │
│ Iteration 1:                                                 │
│   [ErrorDetector] Found 2 errors:                           │
│     - SyntaxError: line 2 (missing parenthesis)             │
│     - NameError: line 5 (process not defined)               │
│                                                               │
│   [DocFetcher] Fetching FastAPI docs...                     │
│   [Qwen3-Coder] Fixing code with documentation context...   │
│   Result: Fixed syntax error                                │
│                                                               │
│ Iteration 2:                                                 │
│   [ErrorDetector] Found 1 error:                            │
│     - NameError: line 5 (process not defined)               │
│   [Qwen3-Coder] Fixing with FastAPI examples...             │
│   Result: Replaced with inline logic                        │
│                                                               │
│ Iteration 3:                                                 │
│   [ErrorDetector] No errors found ✓                         │
│   Code is error-free!                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Returns to Claude:                                           │
│   from fastapi import FastAPI                                │
│                                                               │
│   app = FastAPI()                                            │
│                                                               │
│   @app.get("/chat")                                          │
│   async def chat(msg: str):                                  │
│       return {"response": f"Echo: {msg}"}                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Setup Instructions

### Prerequisites

1. **Python 3.11+** installed
2. **Ollama** running locally with `qwen3-coder:480b-cloud` model
3. **Neo4j** (optional, for Graph RAG features)

### Installation

```bash
# Navigate to the project directory
cd mcp_doc_fetcher

# Install dependencies
pip install -r requirements.txt

# Pull the Qwen3-Coder model in Ollama
ollama pull qwen3-coder:480b-cloud

# Pull the embedding model
ollama pull nomic-embed-text

# Configure environment variables
cp .env.example .env
# Edit .env with your settings
```

### Environment Configuration

Edit `.env` file:

```env
# Ollama Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=qwen3-coder:480b-cloud

# API Key (for web search)
OLLAMA_API_KEY=your_api_key_here

# Cache Settings
CACHE_DB_PATH=./cache/embeddings.db
CACHE_MAX_AGE_HOURS=24

# Server Settings
SERVER_NAME=doc-fetcher-mcp
DEBUG=false
```

---

## 🔌 Integration with Claude Desktop

### Step 1: Add MCP Server to Claude Desktop

Edit your Claude Desktop config file:

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Linux:** `~/.config/Claude/claude_desktop_config.json`

Add this configuration:

```json
{
  "mcpServers": {
    "doc-fetcher": {
      "command": "python",
      "args": [
        "-m",
        "server"
      ],
      "cwd": "C:\\Users\\vedan\\Desktop\\mcp-rag\\contex-king\\context-engineering-intro-main\\mcp_doc_fetcher",
      "env": {
        "PYTHONPATH": "C:\\Users\\vedan\\Desktop\\mcp-rag\\contex-king\\context-engineering-intro-main\\mcp_doc_fetcher"
      }
    }
  }
}
```

**Important:** Replace the `cwd` path with your actual project path.

### Step 2: Restart Claude Desktop

Close and reopen Claude Desktop. The MCP server should connect automatically.

### Step 3: Verify Connection

In Claude Desktop, you should see a "Connected to doc-fetcher" indicator or similar.

Try asking:
```
"Use the validate_and_fix_code tool to check this Python code: [paste code]"
```

---

## 🔌 Integration with VS Code

### Step 1: Install MCP Extension

1. Open VS Code
2. Install the **Model Context Protocol** extension
3. Restart VS Code

### Step 2: Configure MCP Settings

Open VS Code settings (`Ctrl+,` or `Cmd+,`) and search for "MCP".

Add this to your `settings.json`:

```json
{
  "mcp.servers": [
    {
      "name": "doc-fetcher",
      "command": "python",
      "args": ["-m", "server"],
      "cwd": "C:\\Users\\vedan\\Desktop\\mcp-rag\\contex-king\\context-engineering-intro-main\\mcp_doc_fetcher",
      "env": {
        "PYTHONPATH": "C:\\Users\\vedan\\Desktop\\mcp-rag\\contex-king\\context-engineering-intro-main\\mcp_doc_fetcher"
      }
    }
  ]
}
```

### Step 3: Test Integration

1. Open a Python file
2. Right-click → **MCP Tools** → **validate_and_fix_code**
3. Paste your code
4. Watch it auto-fix errors!

---

## 🛠️ Available MCP Tools

### 1. `validate_and_fix_code`

**Purpose:** Automatically validate and fix Python code using documentation.

**Input Schema:**
```json
{
  "code": "string (required) - Python code to validate",
  "project_description": "string (optional) - Project context",
  "max_iterations": "integer (optional, default: 5) - Max fix iterations",
  "python_version": "string (optional, default: '3.11') - Target Python version"
}
```

**Example Usage:**

```python
# In Claude Desktop or VS Code with MCP enabled:

# Ask Claude:
"Use validate_and_fix_code to fix this code:

import requests
def get_data(:
    response = requests.get('https://api.example.com')
    return response.jason()  # Typo: jason instead of json
"
```

**Output:**
```json
{
  "fixed_code": "import requests\n\ndef get_data():\n    response = requests.get('https://api.example.com')\n    return response.json()",
  "libraries_found": ["requests"],
  "iterations": 2,
  "is_error_free": true,
  "fixes_applied": [
    {
      "iteration": 1,
      "errors_before": ["SyntaxError on line 2: invalid syntax"],
      "fix_applied": true
    },
    {
      "iteration": 2,
      "errors_before": ["AttributeError: jason not found"],
      "fix_applied": true
    }
  ]
}
```

### 2. `fetch_documentation`

**Purpose:** Fetch and cache library documentation.

**Input:**
```json
{
  "library_name": "fastapi",
  "version": "0.100.0",  // Optional
  "max_pages": 50,
  "force_refresh": false
}
```

### 3. `search_documentation`

**Purpose:** Search cached docs with semantic similarity.

**Input:**
```json
{
  "library_name": "fastapi",
  "query": "how to add middleware",
  "max_results": 5
}
```

---

## 📊 How It Works Internally

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Server (server.py)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           CodeValidator (code_validator.py)          │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌─────────────┐ │   │
│  │  │  Error     │  │   Ollama     │  │  Version    │ │   │
│  │  │  Detector  │  │   Client     │  │  Resolver   │ │   │
│  │  └────────────┘  └──────────────┘  └─────────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         DocumentCache (cache.py)                     │   │
│  │  - SQLite storage                                     │   │
│  │  - Vector embeddings (nomic-embed-text)              │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         WebSearcher & Crawler (web_search.py)        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
             ┌──────────────────────────────┐
             │  Ollama (Qwen3-Coder 480B)   │
             │  - Code generation            │
             │  - Error fixing               │
             │  - Loop-based regeneration    │
             └──────────────────────────────┘
```

### Loop-Based Fixing Algorithm

```python
async def fix_code_with_loop(code, max_iterations=5):
    current_code = code

    for iteration in range(1, max_iterations + 1):
        # Step 1: Detect errors
        errors = error_detector.detect_all_errors(current_code)

        if not errors:
            return {"fixed_code": current_code, "is_error_free": True}

        # Step 2: Gather documentation
        doc_context = await gather_docs_for_errors(errors)

        # Step 3: Fix with Qwen3-Coder
        fixed_result = await ollama.fix_code_with_context(
            broken_code=current_code,
            error_messages=errors,
            documentation_context=doc_context
        )

        current_code = fixed_result["fixed_code"]

        # Step 4: Check if code changed
        if code_unchanged:
            break

    return {"fixed_code": current_code, "iterations": iteration}
```

---

## 🧪 Testing

### Run Unit Tests

```bash
cd mcp_doc_fetcher
python -m pytest tests/test_code_validator.py -v
```

### Run Integration Test

```bash
python -m pytest tests/test_integration_end_to_end.py -v -s
```

Expected output:
```
======================================================================
INTEGRATION TEST: End-to-End Code Validation Workflow
======================================================================

[Step 1] Initializing CodeValidator...
[Step 2] Claude generated code (with errors):
----------------------------------------------------------------------
import fastapi
from fastapi import FastAPI
app = FastAPI(
...
----------------------------------------------------------------------

[Step 3] Running code validation and auto-fix...
[Step 4] Validation Results:
  - Libraries found: ['fastapi']
  - Iterations taken: 2
  - Error-free: True
  - Fixes applied: 2

✓ END-TO-END TEST PASSED
```

---

## 🔧 Troubleshooting

### Issue: "Ollama connection refused"

**Solution:**
```bash
# Check if Ollama is running
ollama list

# Start Ollama if not running
ollama serve

# Verify model is available
ollama pull qwen3-coder:480b-cloud
```

### Issue: "No module named 'mcp'"

**Solution:**
```bash
pip install --upgrade mcp
```

### Issue: "Documentation not found"

**Solution:**
```bash
# Clear cache and refetch
# In Python/Claude:
# Use tool: clear_cache
# Then: fetch_documentation with force_refresh=True
```

### Issue: "Code still has errors after 5 iterations"

**Possible causes:**
1. Documentation not cached → Run `fetch_documentation` first
2. Complex errors → Increase `max_iterations` to 10
3. Model not responding → Check Ollama logs

---

## 💡 Usage Examples

### Example 1: Fix Broken Flask App

**Input to Claude:**
```
"Use validate_and_fix_code to fix this Flask code:

from flask import Flask
app = Flask(__name__

@app.route('/hello')
def hello(
    return 'Hello World'
"
```

**Output:** Error-free Flask app with correct syntax.

### Example 2: Fix Import Errors

**Input:**
```python
import requests
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/fetch")
async def fetch_data():
    resp = requests.get("https://api.github.com")
    return resp.json
```

**Auto-fixes:**
- Adds `()` to `resp.json()` (method call missing)
- Validates `requests` compatibility with `async` functions

### Example 3: Version Compatibility Check

**Input:**
```python
import pandas as pd
import numpy as np
from typing import List

def process_data(data: List[float]):
    df = pd.DataFrame(data)
    return df.describe()
```

**Output:** Checks if pandas/numpy versions are compatible with Python 3.11.

---

## 🎯 Best Practices

### 1. Pre-fetch Common Libraries

Before starting a project, fetch docs for common libraries:

```python
# Ask Claude to run these:
fetch_documentation(library_name="fastapi")
fetch_documentation(library_name="pydantic")
fetch_documentation(library_name="sqlalchemy")
```

### 2. Use Project Descriptions

Provide context for better fixes:

```python
validate_and_fix_code(
    code=my_code,
    project_description="REST API with JWT authentication and PostgreSQL"
)
```

### 3. Increase Iterations for Complex Code

For large files or complex errors:

```python
validate_and_fix_code(code=my_code, max_iterations=10)
```

---

## 🚀 Advanced: Automatic Triggering

### Option A: Git Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Auto-fix Python files before commit

for file in $(git diff --cached --name-only --diff-filter=ACM | grep '\.py$'); do
    echo "Validating $file..."
    python -c "
import asyncio
from utils.code_validator import CodeValidator

async def fix_file(path):
    with open(path) as f:
        code = f.read()

    validator = CodeValidator(...)
    result = await validator.validate_and_fix_code(code)

    if not result['is_error_free']:
        print(f'⚠️  {path} still has errors')
        exit(1)

    if result['fixed_code'] != code:
        with open(path, 'w') as f:
            f.write(result['fixed_code'])
        print(f'✓ Fixed {path}')

asyncio.run(fix_file('$file'))
"
done
```

### Option B: VS Code Extension (Custom)

Create a VS Code extension that:
1. Listens to file save events
2. Calls MCP `validate_and_fix_code` tool
3. Applies fixes automatically

---

## 📚 References

- **MCP Protocol:** https://modelcontextprotocol.io/
- **Ollama API:** https://ollama.ai/docs
- **Qwen3-Coder:** https://huggingface.co/Qwen/Qwen3-Coder

---

## 📄 License

MIT License

---

## 🤝 Support

For issues or questions:
- GitHub Issues: [Your repo URL]
- Email: [Your email]

---

**Congratulations!** 🎉 You now have a universal AI code validator that works with Claude Desktop, VS Code, and any MCP-compatible IDE, using local Ollama models to save Claude tokens while delivering error-free code.
