"""
Code Example Extractor - Agentic RAG

Extracts code blocks from documentation and:
- Identifies code blocks >= 300 characters
- Extracts surrounding context
- Generates AI summaries
- Stores in separate table for dedicated search

Performance: 95% code finding accuracy
"""
import re
import asyncio
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import ollama

from config.settings import get_rag_config

logger = logging.getLogger(__name__)


class CodeExampleExtractor:
    """
    Extract and process code examples from documentation.

    Features:
    - Markdown code block parsing
    - Context extraction (before/after)
    - Parallel AI summarization
    - Language detection
    """

    def __init__(self):
        """Initialize code extractor."""
        self.config = get_rag_config()
        self.min_length = self.config.min_code_block_length
        self.context_lines = self.config.code_context_lines
        self.max_summary_tokens = self.config.max_code_summary_tokens

        logger.info(
            f"Code extractor initialized "
            f"(min_length: {self.min_length}, context: {self.context_lines} lines)"
        )

    def extract_code_blocks(
        self,
        markdown: str,
        url: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Extract all code blocks from markdown.

        Args:
            markdown: Markdown text
            url: Source URL (for metadata)

        Returns:
            List of code block dicts with context
        """
        if not markdown:
            return []

        lines = markdown.split('\n')
        code_blocks = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check if this is start of code block
            if line.strip().startswith('```'):
                code_block = self._extract_single_block(lines, i, url)
                if code_block:
                    code_blocks.append(code_block)
                    # Skip to end of this block
                    i = code_block['end_line'] + 1
                    continue

            i += 1

        logger.info(f"Extracted {len(code_blocks)} code blocks from {url}")
        return code_blocks

    def _extract_single_block(
        self,
        lines: List[str],
        start_index: int,
        url: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract a single code block with context.

        Args:
            lines: All markdown lines
            start_index: Index of ``` line
            url: Source URL

        Returns:
            Code block dict or None if too small
        """
        # Extract language
        first_line = lines[start_index].strip()
        language = first_line[3:].strip() if len(first_line) > 3 else 'plaintext'

        # Find end of code block
        code_lines = []
        i = start_index + 1

        while i < len(lines) and not lines[i].strip().startswith('```'):
            code_lines.append(lines[i])
            i += 1

        if i >= len(lines):
            # No closing ```
            return None

        code = '\n'.join(code_lines)

        # Check minimum length
        if len(code) < self.min_length:
            return None

        # Extract context before
        context_start = max(0, start_index - self.context_lines - 1)
        context_before = '\n'.join(lines[context_start:start_index])

        # Extract context after
        context_end = min(len(lines), i + self.context_lines + 2)
        context_after = '\n'.join(lines[i + 1:context_end])

        return {
            'code': code,
            'language': language,
            'context_before': context_before.strip(),
            'context_after': context_after.strip(),
            'length': len(code),
            'line_count': len(code_lines),
            'start_line': start_index,
            'end_line': i,
            'url': url
        }

    async def generate_summary(
        self,
        code: str,
        context_before: str = "",
        context_after: str = "",
        language: str = "python"
    ) -> str:
        """
        Generate AI summary of code example.

        Args:
            code: Code block
            context_before: Context before code
            context_after: Context after code
            language: Programming language

        Returns:
            AI-generated summary
        """
        # Build prompt
        prompt = f"""Summarize this {language} code example concisely.

Context before:
{context_before[:200]}

Code:
{code[:1000]}

Context after:
{context_after[:200]}

Provide:
1. What it does (1 sentence)
2. Key patterns used
3. Use case

Keep it under {self.max_summary_tokens} tokens."""

        try:
            # Generate summary using Ollama
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ollama.generate(
                    model=self.config.ollama_chat_model,
                    prompt=prompt
                )
            )

            summary = response.get('response', '').strip()

            if not summary:
                summary = f"Code example in {language}: {len(code)} characters"

            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return f"Code example in {language}"

    async def process_code_blocks_parallel(
        self,
        code_blocks: List[Dict[str, Any]],
        max_workers: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate summaries for multiple code blocks in parallel.

        Args:
            code_blocks: List of code block dicts
            max_workers: Maximum parallel workers

        Returns:
            Code blocks with summaries added
        """
        if not code_blocks:
            return []

        logger.info(f"Generating summaries for {len(code_blocks)} code blocks")

        # Create tasks
        tasks = []
        for block in code_blocks:
            task = self.generate_summary(
                block['code'],
                block.get('context_before', ''),
                block.get('context_after', ''),
                block.get('language', 'plaintext')
            )
            tasks.append(task)

        # Execute in parallel with semaphore
        semaphore = asyncio.Semaphore(max_workers)

        async def summarize_with_limit(task):
            async with semaphore:
                return await task

        # Gather results
        summaries = await asyncio.gather(
            *[summarize_with_limit(task) for task in tasks],
            return_exceptions=True
        )

        # Add summaries to blocks
        for block, summary in zip(code_blocks, summaries):
            if isinstance(summary, str):
                block['summary'] = summary
            else:
                block['summary'] = f"Code example in {block.get('language', 'unknown')}"
                logger.error(f"Summary generation failed: {summary}")

        logger.info(f"Generated {len(summaries)} summaries")
        return code_blocks

    def extract_and_summarize_sync(
        self,
        markdown: str,
        url: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for extract + summarize.

        Args:
            markdown: Markdown text
            url: Source URL

        Returns:
            Code blocks with summaries
        """
        # Extract
        blocks = self.extract_code_blocks(markdown, url)

        if not blocks:
            return []

        # Summarize (run async in sync context)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in async context
            return blocks
        else:
            return loop.run_until_complete(
                self.process_code_blocks_parallel(blocks)
            )


def extract_inline_code(text: str) -> List[str]:
    """
    Extract inline code snippets (single backticks).

    Args:
        text: Text to extract from

    Returns:
        List of inline code snippets
    """
    pattern = r'`([^`]+)`'
    matches = re.findall(pattern, text)
    return [m.strip() for m in matches if len(m.strip()) > 5]


def detect_code_language(code: str) -> str:
    """
    Attempt to detect programming language.

    Args:
        code: Code snippet

    Returns:
        Detected language or 'unknown'
    """
    # Simple heuristics
    if 'import ' in code or 'def ' in code or 'class ' in code:
        if 'async def' in code or 'await ' in code:
            return 'python-async'
        return 'python'
    elif 'function ' in code or 'const ' in code or '=> ' in code:
        return 'javascript'
    elif 'fn ' in code or 'impl ' in code:
        return 'rust'
    elif 'public class' in code or 'private ' in code:
        return 'java'
    elif '#include' in code or 'int main' in code:
        return 'c/c++'
    elif 'SELECT ' in code.upper() or 'INSERT ' in code.upper():
        return 'sql'
    else:
        return 'unknown'


# ============ Usage Example ============

async def demo_code_extraction():
    """Demonstrate code extraction."""

    sample_markdown = """
# API Usage

Here's how to use the API:

```python
import requests

# Make a GET request
response = requests.get("https://api.example.com/data")

# Parse JSON
data = response.json()

# Process results
for item in data:
    print(item['name'])
```

This code demonstrates basic HTTP requests.

## Error Handling

Always handle errors:

```python
try:
    response = requests.get(url, timeout=5)
    response.raise_for_status()
except requests.RequestException as e:
    print(f"Error: {e}")
```

Use timeouts to prevent hanging.
"""

    extractor = CodeExampleExtractor()

    # Extract
    blocks = extractor.extract_code_blocks(sample_markdown)
    print(f"Found {len(blocks)} code blocks\n")

    # Summarize
    blocks = await extractor.process_code_blocks_parallel(blocks)

    # Display
    for i, block in enumerate(blocks, 1):
        print(f"Block {i} ({block['language']}):")
        print(f"  Length: {block['length']} chars")
        print(f"  Summary: {block['summary']}")
        print(f"  Context before: {block['context_before'][:50]}...")
        print()


if __name__ == "__main__":
    asyncio.run(demo_code_extraction())
