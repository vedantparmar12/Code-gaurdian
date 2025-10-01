# System Architecture Diagram

## Complete System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERACTION                                │
│                                                                               │
│  User: "Create a chatbot with FastAPI"                                      │
│    ↓                                                                         │
│  Claude Desktop / VS Code (MCP Client)                                      │
│    ↓                                                                         │
│  Claude Code generates:                                                     │
│    import fastapi                                                           │
│    app = FastAPI(  # ← Missing )                                           │
│    @app.get("/chat")                                                        │
│    def chat(msg):                                                           │
│        return process(msg)  # ← Undefined function                         │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    │ MCP Protocol
                                    │ call validate_and_fix_code
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MCP SERVER (server.py)                              │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                  TOOL: validate_and_fix_code                           │ │
│  │                                                                         │ │
│  │  Input:                                                                 │ │
│  │    - code: str (broken code from Claude)                               │ │
│  │    - project_description: str (optional)                               │ │
│  │    - max_iterations: int = 5                                           │ │
│  │    - python_version: str = "3.11"                                      │ │
│  └────────────────────────────────┬────────────────────────────────────────┘ │
│                                   │                                          │
│                                   ↓                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │              CODE VALIDATOR (utils/code_validator.py)                  │ │
│  │                                                                         │ │
│  │  Step 1: Extract Imports                                               │ │
│  │    ├─ AST parsing                                                      │ │
│  │    ├─ Filter standard library                                          │ │
│  │    └─ Result: ["fastapi"]                                              │ │
│  │                                                                         │ │
│  │  Step 2: Fetch Documentation                                           │ │
│  │    ├─ Check cache first ──────────────────┐                            │ │
│  │    ├─ If not cached:                      │                            │ │
│  │    │   ├─ PyPI API                        │                            │ │
│  │    │   ├─ Web crawling                    │                            │ │
│  │    │   └─ Store in cache                  │                            │ │
│  │    └─ Result: FastAPI docs fetched        │                            │ │
│  │                                            │                            │ │
│  │  Step 3: Version Compatibility ───────────┼───────┐                    │ │
│  │    ├─ Query PyPI API                      │       │                    │ │
│  │    ├─ Check Python version                │       │                    │ │
│  │    ├─ Check dependencies                  │       │                    │ │
│  │    └─ Result: All compatible              │       │                    │ │
│  │                                            │       │                    │ │
│  │  Step 4: Detect Errors ────────────────────┤       │                    │ │
│  │    ├─ ErrorDetector.detect_all_errors()   │       │                    │ │
│  │    │   ├─ Syntax check (AST)              │       │                    │ │
│  │    │   ├─ Import check                    │       │                    │ │
│  │    │   ├─ Undefined names                 │       │                    │ │
│  │    │   └─ Common mistakes                 │       │                    │ │
│  │    └─ Result: [SyntaxError, NameError]    │       │                    │ │
│  │                                            │       │                    │ │
│  │  Step 5: Loop-Based Fixing ────────────────┘       │                    │ │
│  │    ┌────────────────────────────────────┐          │                    │ │
│  │    │   for iteration in 1..max_iter:    │          │                    │ │
│  │    │                                    │          │                    │ │
│  │    │   ┌─ Detect errors                │          │                    │ │
│  │    │   │  If none: RETURN fixed_code   │          │                    │ │
│  │    │   │                                │          │                    │ │
│  │    │   ├─ Gather documentation context │◄─────────┘                    │ │
│  │    │   │  └─ Search docs for errors    │                               │ │
│  │    │   │                                │                               │ │
│  │    │   ├─ Call Qwen3-Coder ─────────────┼────────────┐                 │ │
│  │    │   │  └─ Fix with context          │            │                 │ │
│  │    │   │                                │            │                 │ │
│  │    │   └─ Update current_code          │            │                 │ │
│  │    │                                    │            │                 │ │
│  │    └────────────────────────────────────┘            │                 │ │
│  │                                                       │                 │ │
│  └───────────────────────────────────────────────────────┼─────────────────┘ │
│                                                          │                   │
└──────────────────────────────────────────────────────────┼───────────────────┘
                                                           │
                                                           │
┌──────────────────────────────────────────────────────────┼───────────────────┐
│                      SUPPORTING COMPONENTS               │                   │
│                                                          │                   │
│  ┌────────────────────────┐  ┌────────────────────────┐ │                   │
│  │   ErrorDetector        │  │   VersionResolver      │ │                   │
│  │   (error_detector.py)  │  │   (version_resolver.py)│ │                   │
│  │                        │  │                        │ │                   │
│  │  - detect_all_errors() │  │  - get_package_info()  │ │                   │
│  │  - check_syntax()      │  │  - check_compatibility()│ │                  │
│  │  - check_imports()     │  │  - resolve_dependencies│ │                   │
│  │  - check_undefined()   │  │  - suggest_versions()  │ │                   │
│  │  - check_mistakes()    │  │                        │ │                   │
│  │                        │  │  PyPI API ──────────────┼─────► pypi.org    │
│  └────────────────────────┘  └────────────────────────┘ │                   │
│                                                          │                   │
│  ┌──────────────────────────────────────────────────────┴────────────────┐  │
│  │                   OllamaClient (ollama_client.py)                     │  │
│  │                                                                        │  │
│  │  - fix_code_with_context(broken_code, errors, docs)                  │  │
│  │  - generate_code(prompt)                                              │  │
│  │  - chat(messages)                                                     │  │
│  │  - extract_code_block(response)                                       │  │
│  │  - check_syntax(code)                                                 │  │
│  │                                                                        │  │
│  │  HTTP POST ──────────────────────────────────────────────────────────┼──┐│
│  └──────────────────────────────────────────────────────────────────────┘  ││
│                                                                              ││
│  ┌────────────────────────────────────────────────────────────────────────┐││
│  │              DocumentCache (utils/cache.py)                            │││
│  │                                                                         │││
│  │  SQLite Database: ./cache/embeddings.db                                │││
│  │  ├─ libraries table (metadata)                                         │││
│  │  ├─ pages table (documentation content)                                │││
│  │  └─ embeddings table (vector search)                                   │││
│  │                                                                         │││
│  │  Methods:                                                               │││
│  │  ├─ get_library_documentation(name)                                    │││
│  │  ├─ store_library_documentation(docs)                                  │││
│  │  └─ search_embeddings(query_vector)                                    │││
│  └────────────────────────────────────────────────────────────────────────┘││
│                                                                              ││
│  ┌────────────────────────────────────────────────────────────────────────┐││
│  │         WebSearcher & Crawler (web_search.py, crawler.py)              │││
│  │                                                                         │││
│  │  - search_documentation_urls(library_name)                             │││
│  │  - crawl_pages_concurrent(urls)                                        │││
│  │                                                                         │││
│  │  PyPI API ──────────────────────────────► pypi.org                     │││
│  │  DuckDuckGo ─────────────────────────────► duckduckgo.com             │││
│  └────────────────────────────────────────────────────────────────────────┘││
│                                                                              ││
└──────────────────────────────────────────────────────────────────────────────┘│
                                                                                │
                                                                                │
┌───────────────────────────────────────────────────────────────────────────────┘
│                        OLLAMA SERVER (Local)
│
│  ┌────────────────────────────────────────────────────────────────────────┐
│  │                    http://localhost:11434                              │
│  │                                                                         │
│  │  Models:                                                                │
│  │  ├─ qwen3-coder:480b-cloud  (Code generation & fixing)                │
│  │  └─ nomic-embed-text         (Semantic embeddings)                    │
│  │                                                                         │
│  │  Endpoints:                                                             │
│  │  ├─ POST /api/generate  (Text generation)                             │
│  │  ├─ POST /api/chat      (Conversational)                              │
│  │  └─ POST /api/embeddings (Vector embeddings)                          │
│  └────────────────────────────────────────────────────────────────────────┘
│
└────────────────────────────────────────────────────────────────────────────────


                              RETURN PATH
                              ───────────

  Fixed Code (Error-Free) ───────────────────────────────┐
                                                          │
  {                                                       │
    "fixed_code": "from fastapi import FastAPI\n...",    │
    "libraries_found": ["fastapi"],                      │
    "iterations": 2,                                     │
    "is_error_free": true,                               │
    "fixes_applied": [...]                               │
  }                                                       │
                                                          │
                    ┌─────────────────────────────────────┘
                    │
                    ↓
         ┌──────────────────────────┐
         │  MCP Server Response     │
         │  (CallToolResult)        │
         └─────────────┬────────────┘
                       │
                       ↓
         ┌──────────────────────────┐
         │  Claude Desktop/VS Code  │
         │  (MCP Client)            │
         └─────────────┬────────────┘
                       │
                       ↓
         ┌──────────────────────────┐
         │  User sees fixed code    │
         │  ✅ No errors            │
         │  💰 Zero tokens used     │
         └──────────────────────────┘
```

---

## Data Flow: Single Iteration

```
┌────────────────────────────────────────────────────────────────┐
│  Iteration 1: Fix Syntax Error                                 │
└────────────────────────────────────────────────────────────────┘

Input Code:
┌──────────────────────────┐
│ app = FastAPI(           │  ← Missing closing parenthesis
│ @app.get("/chat")        │
│ def chat(): pass         │
└──────────────────────────┘
         │
         ↓
┌──────────────────────────────────────────────────────────────┐
│ ErrorDetector.detect_all_errors()                            │
│                                                               │
│ Result:                                                       │
│ [                                                             │
│   {                                                           │
│     "type": "SyntaxError",                                   │
│     "message": "invalid syntax",                             │
│     "line": 1,                                               │
│     "severity": "error"                                      │
│   }                                                           │
│ ]                                                             │
└──────────────────────────────┬────────────────────────────────┘
                               │
                               ↓
┌──────────────────────────────────────────────────────────────┐
│ Gather Documentation Context                                 │
│                                                               │
│ - Search "fastapi" docs for "getting started"                │
│ - Search "fastapi" docs for "invalid syntax"                 │
│                                                               │
│ Result:                                                       │
│ """                                                           │
│ === FastAPI Documentation ===                                │
│                                                               │
│ [Example 1]:                                                  │
│ from fastapi import FastAPI                                  │
│                                                               │
│ app = FastAPI()  # ← Correct syntax                         │
│                                                               │
│ @app.get("/")                                                │
│ async def root():                                            │
│     return {"message": "Hello World"}                        │
│ """                                                           │
└──────────────────────────────┬────────────────────────────────┘
                               │
                               ↓
┌──────────────────────────────────────────────────────────────┐
│ OllamaClient.fix_code_with_context()                         │
│                                                               │
│ Prompt to Qwen3-Coder:                                       │
│ """                                                           │
│ Fix the following Python code that has errors:               │
│                                                               │
│ **Errors Found:**                                            │
│ - [SyntaxError] Line 1: invalid syntax                       │
│                                                               │
│ **Broken Code:**                                             │
│ ```python                                                    │
│ app = FastAPI(                                               │
│ @app.get("/chat")                                            │
│ def chat(): pass                                             │
│ ```                                                          │
│                                                               │
│ **Official Documentation Reference:**                        │
│ [FastAPI docs showing correct usage...]                      │
│                                                               │
│ **Instructions:**                                            │
│ Fix ALL errors and return corrected code.                    │
│ """                                                           │
│                                                               │
│ Qwen3-Coder Response:                                        │
│ ```python                                                    │
│ from fastapi import FastAPI                                  │
│                                                               │
│ app = FastAPI()  # ← FIXED                                  │
│                                                               │
│ @app.get("/chat")                                            │
│ def chat():                                                  │
│     pass                                                     │
│ ```                                                          │
└──────────────────────────────┬────────────────────────────────┘
                               │
                               ↓
┌──────────────────────────────────────────────────────────────┐
│ Updated Code:                                                │
│ ┌──────────────────────────┐                                │
│ │ from fastapi import...   │                                │
│ │ app = FastAPI()          │  ← Fixed!                      │
│ │ @app.get("/chat")        │                                │
│ │ def chat(): pass         │                                │
│ └──────────────────────────┘                                │
└──────────────────────────────────────────────────────────────┘

Next iteration will check this fixed code for remaining errors...
```

---

## Token Cost Comparison

```
┌──────────────────────────────────────────────────────────────┐
│              WITHOUT MCP Auto-Fix                             │
│                                                               │
│  1. Claude generates broken code                             │
│     Cost: ~1000 tokens                                       │
│                                                               │
│  2. User: "This has an error, please fix"                    │
│     Cost: ~50 tokens                                         │
│                                                               │
│  3. Claude analyzes and regenerates                          │
│     Cost: ~1500 tokens                                       │
│                                                               │
│  4. Still has error, user asks again                         │
│     Cost: ~50 tokens                                         │
│                                                               │
│  5. Claude fixes again                                       │
│     Cost: ~1500 tokens                                       │
│                                                               │
│  TOTAL: ~4100 tokens = $0.05-0.10 USD                       │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│               WITH MCP Auto-Fix                               │
│                                                               │
│  1. Claude generates broken code                             │
│     Cost: ~1000 tokens                                       │
│                                                               │
│  2. MCP validate_and_fix_code (automatic)                    │
│     Cost: 0 tokens (uses local Ollama)                      │
│     ├─ Fetch docs: Local cache                              │
│     ├─ Detect errors: Local AST                             │
│     ├─ Fix code: Local Qwen3-Coder                          │
│     └─ Iterations: All local                                │
│                                                               │
│  3. Returns fixed code to Claude                             │
│     Cost: 0 tokens (MCP tool result)                        │
│                                                               │
│  TOTAL: ~1000 tokens = $0.01-0.02 USD                       │
│                                                               │
│  SAVINGS: 75% reduction in token costs! 💰                   │
└──────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
mcp_doc_fetcher/
│
├── server.py                      # MCP server entry point
├── config.py                      # Configuration management
├── models.py                      # Pydantic models
├── requirements.txt               # Dependencies
├── .env                           # Environment variables
│
├── utils/
│   ├── __init__.py
│   ├── ollama_client.py          # NEW: Ollama API integration
│   ├── error_detector.py         # NEW: Error detection
│   ├── version_resolver.py       # NEW: Version compatibility
│   ├── code_validator.py         # ENHANCED: Loop-based fixing
│   ├── cache.py                  # SQLite caching
│   ├── crawler.py                # Web crawling
│   ├── embeddings.py             # Semantic search
│   └── web_search.py             # Documentation search
│
├── tests/
│   ├── __init__.py
│   ├── test_code_validator.py          # NEW: Unit tests
│   └── test_integration_end_to_end.py  # NEW: Integration test
│
└── docs/
    ├── README.md                  # UPDATED: Project overview
    ├── QUICKSTART.md              # NEW: Quick start guide
    ├── INTEGRATION_GUIDE.md       # NEW: Full integration guide
    ├── IMPLEMENTATION_SUMMARY.md  # NEW: Technical summary
    └── SYSTEM_DIAGRAM.md          # NEW: This file
```

---

## Key Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| **MCP** | Tool protocol | 1.0.0+ |
| **Ollama** | Local LLM server | Latest |
| **Qwen3-Coder** | Code generation | 480B (or 7B) |
| **nomic-embed-text** | Embeddings | Latest |
| **Python** | Implementation | 3.11+ |
| **aiohttp** | Async HTTP | 3.9+ |
| **SQLite** | Caching | Built-in |
| **AST** | Parsing | Built-in |
| **PyPI API** | Package metadata | REST |
| **pytest** | Testing | 7.0+ |

---

**System Status:** ✅ **PRODUCTION READY**

All components implemented, tested, and documented.
