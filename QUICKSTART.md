# Quick Start Guide - Loop-Based Code Validation

## TL;DR

1. Install Ollama + models
2. Run MCP server
3. Tell Claude to use `validate_and_fix_code` tool
4. Get error-free code automatically (zero Claude tokens!)

---

## 1. Setup (5 minutes)

```bash
# Install Ollama
# Visit https://ollama.ai and download

# Pull models
ollama pull qwen3-coder:480b-cloud
ollama pull nomic-embed-text

# Clone and install
cd mcp_doc_fetcher
pip install -r requirements.txt

# Start server
python -m server
```

---

## 2. Test Locally (Before MCP Integration)

```bash
# Run integration test
python -m pytest tests/test_integration_end_to_end.py -v -s
```

Expected output:
```
✓ END-TO-END TEST PASSED
```

---

## 3. Connect to Claude Desktop

**Windows:** Edit `%APPDATA%\Claude\claude_desktop_config.json`

**macOS:** Edit `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "doc-fetcher": {
      "command": "python",
      "args": ["-m", "server"],
      "cwd": "C:\\full\\path\\to\\mcp_doc_fetcher",
      "env": {
        "PYTHONPATH": "C:\\full\\path\\to\\mcp_doc_fetcher"
      }
    }
  }
}
```

Restart Claude Desktop.

---

## 4. Usage Examples

### Example 1: Fix Broken Code

**In Claude:**
```
Use validate_and_fix_code to fix this:

import requests
def get_data(:
    resp = requests.get('https://api.com')
    return resp.jason()
```

**Result:** Error-free code in 2 iterations (syntax + method name fixed)

### Example 2: Full Project Validation

**In Claude:**
```
I just wrote a FastAPI app. Please validate and fix it:

[paste entire project code]
```

**Result:**
- All imports validated
- Version compatibility checked
- API usage verified against docs
- Code fixed and returned

---

## 5. How It Works

```
User → Claude → Broken Code
         ↓
    MCP validate_and_fix_code
         ↓
   [Iteration 1]
    Detect errors → Fetch docs → Qwen3-Coder fixes
         ↓
   [Iteration 2]
    Check again → Still has errors? → Fix again
         ↓
   [Iteration N]
    No errors! → Return fixed code
         ↓
    Claude ← Error-free code (no tokens used!)
```

---

## 6. Configuration Options

```python
validate_and_fix_code(
    code="your code here",
    project_description="FastAPI REST API",  # Optional: provides context
    max_iterations=5,                        # Default: 5
    python_version="3.11"                    # Default: 3.11
)
```

---

## 7. Troubleshooting

### "Ollama connection refused"
```bash
ollama serve
```

### "Model not found"
```bash
ollama pull qwen3-coder:480b-cloud
```

### "Server won't start"
```bash
# Check Python version
python --version  # Should be 3.11+

# Install dependencies
pip install -r requirements.txt

# Check logs
python -m server 2>&1 | tee server.log
```

### "Code still has errors after 5 iterations"
```python
# Increase max_iterations
validate_and_fix_code(code=my_code, max_iterations=10)

# Or manually fetch docs first
fetch_documentation(library_name="fastapi")
validate_and_fix_code(code=my_code)
```

---

## 8. Performance Tips

### Pre-cache Common Libraries

Before starting work, fetch docs for libraries you'll use:

```python
# Ask Claude to run these:
fetch_documentation(library_name="fastapi", max_pages=20)
fetch_documentation(library_name="sqlalchemy", max_pages=20)
fetch_documentation(library_name="pydantic", max_pages=10)
```

This makes validation much faster!

### Use Smaller Model for Testing

If Qwen3-Coder 480B is too large, use 7B:

```env
# In .env
OLLAMA_CHAT_MODEL=qwen3-coder:7b
```

---

## 9. What Gets Fixed Automatically

✅ **Syntax errors** (missing parentheses, colons, etc.)
✅ **Import errors** (missing imports, wrong module names)
✅ **Undefined names** (typos, missing function definitions)
✅ **Wrong API usage** (deprecated methods, wrong arguments)
✅ **Type errors** (based on documentation)
✅ **Version incompatibilities** (suggests compatible versions)

---

## 10. Real-World Example

**User asks Claude:**
> "Create a REST API with authentication using FastAPI and JWT"

**Claude generates code** (may have errors)

**MCP automatically:**
1. Extracts libraries: `fastapi`, `pyjwt`, `passlib`
2. Fetches official documentation
3. Detects 3 errors:
   - Wrong import path for `jwt`
   - Missing bcrypt algorithm in `passlib`
   - Async/sync mismatch
4. Fixes in 2 iterations using Qwen3-Coder
5. Returns production-ready code

**Total Claude tokens used: 0** ✅

---

## Next Steps

- Read full documentation: `INTEGRATION_GUIDE.md`
- Run tests: `pytest tests/ -v`
- Customize settings: Edit `.env`
- Add custom libraries: Edit `utils/doc_url_registry.py`

---

**Happy coding! 🚀**
