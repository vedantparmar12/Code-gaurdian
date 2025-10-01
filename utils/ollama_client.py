"""Ollama client for code generation and analysis."""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
import aiohttp

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API for code generation."""

    def __init__(self, base_url: str, model: str):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server URL
            model: Model name to use for generation
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Create aiohttp session."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()

    async def generate_code(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4000
    ) -> Optional[str]:
        """
        Generate code using Ollama model.

        Args:
            prompt: User prompt for code generation
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated code as string, or None on error
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }

            if system_prompt:
                payload["system"] = system_prompt

            logger.info(f"Generating code with {self.model}...")

            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama API error: {response.status} - {error_text}")
                    return None

                result = await response.json()
                generated_text = result.get("response", "").strip()

                logger.info(f"Generated {len(generated_text)} characters of code")
                return generated_text

        except asyncio.TimeoutError:
            logger.error("Ollama request timed out")
            return None
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return None

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4000
    ) -> Optional[str]:
        """
        Chat with Ollama model using conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Response text, or None on error
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }

            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama chat error: {response.status} - {error_text}")
                    return None

                result = await response.json()
                message = result.get("message", {})
                content = message.get("content", "").strip()

                return content

        except asyncio.TimeoutError:
            logger.error("Ollama chat request timed out")
            return None
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return None

    async def extract_code_block(self, text: str) -> str:
        """
        Extract code from markdown code blocks.

        Args:
            text: Text potentially containing code blocks

        Returns:
            Extracted code
        """
        # Try to extract from markdown code blocks
        import re

        # Match ```python ... ``` or ``` ... ```
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            # Return the first code block
            return matches[0].strip()

        # If no code blocks, return the text as-is
        return text.strip()

    async def fix_code_with_context(
        self,
        broken_code: str,
        error_messages: List[str],
        documentation_context: str,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Fix code iteratively using documentation context.

        Args:
            broken_code: Code with errors
            error_messages: List of error messages
            documentation_context: Relevant documentation snippets
            max_attempts: Maximum fix attempts

        Returns:
            Dict with fixed code and metadata
        """
        conversation_history = []
        current_code = broken_code

        system_prompt = """You are an expert Python code fixer. Your job is to:
1. Analyze the broken code and error messages
2. Use the provided official documentation to understand correct usage
3. Fix the code to be error-free and follow best practices
4. Return ONLY the fixed Python code in a code block

IMPORTANT: Always check the documentation for:
- Correct function signatures
- Required vs optional parameters
- Correct import statements
- Correct API usage patterns
- Version-specific changes"""

        for attempt in range(max_attempts):
            logger.info(f"Fix attempt {attempt + 1}/{max_attempts}")

            # Build prompt with error context
            error_context = "\n".join([f"- {err}" for err in error_messages])

            user_prompt = f"""Fix the following Python code that has errors:

**Errors Found:**
{error_context}

**Broken Code:**
```python
{current_code}
```

**Official Documentation Reference:**
{documentation_context}

**Instructions:**
1. Read the error messages carefully
2. Consult the documentation to find correct usage
3. Fix ALL errors in the code
4. Return the complete corrected code in a Python code block
5. Do not add explanations, only return the fixed code"""

            conversation_history.append({
                "role": "user",
                "content": user_prompt
            })

            # Get fix from model
            response = await self.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    *conversation_history
                ],
                temperature=0.1
            )

            if not response:
                logger.error(f"Failed to get response on attempt {attempt + 1}")
                continue

            # Extract code from response
            fixed_code = await self.extract_code_block(response)

            # Add assistant response to history
            conversation_history.append({
                "role": "assistant",
                "content": response
            })

            # Check if code is syntactically valid
            syntax_errors = self._check_syntax(fixed_code)

            if not syntax_errors:
                logger.info(f"Code fixed successfully on attempt {attempt + 1}")
                return {
                    "success": True,
                    "fixed_code": fixed_code,
                    "attempts": attempt + 1,
                    "conversation": conversation_history
                }

            # Update for next iteration
            current_code = fixed_code
            error_messages = syntax_errors
            logger.info(f"Still has errors, retrying... ({len(syntax_errors)} errors)")

        # Max attempts reached
        logger.warning(f"Failed to fix code after {max_attempts} attempts")
        return {
            "success": False,
            "fixed_code": current_code,
            "attempts": max_attempts,
            "remaining_errors": error_messages,
            "conversation": conversation_history
        }

    def _check_syntax(self, code: str) -> List[str]:
        """
        Check Python syntax and return error messages.

        Args:
            code: Python code to check

        Returns:
            List of error messages (empty if no errors)
        """
        import ast

        errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"SyntaxError on line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"ParseError: {str(e)}")

        return errors
