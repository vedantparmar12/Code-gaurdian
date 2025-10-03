"""Optimized string matching using Aho-Corasick algorithm for security validation."""

from typing import List, Dict, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class TrieNode:
    """Node in the Aho-Corasick trie."""

    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.fail: 'TrieNode' = None
        self.output: List[str] = []


class AhoCorasickMatcher:
    """
    Aho-Corasick automaton for fast multi-pattern string matching.

    Use Cases:
    - Finding deprecated functions in code (e.g., 10,000+ patterns)
    - Security scanning for forbidden APIs
    - Detecting vulnerable library versions

    Performance: O(n + m + z) where:
    - n = text length
    - m = total pattern length
    - z = number of matches

    Much faster than regex unions for large pattern sets.
    """

    def __init__(self):
        self.root = TrieNode()
        self.patterns_count = 0

    def add_pattern(self, pattern: str) -> None:
        """Add a pattern to the matcher."""
        node = self.root
        for char in pattern:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.output.append(pattern)
        self.patterns_count += 1

    def add_patterns(self, patterns: List[str]) -> None:
        """Add multiple patterns at once."""
        for pattern in patterns:
            self.add_pattern(pattern)
        logger.info(f"Added {len(patterns)} patterns to Aho-Corasick matcher")

    def build(self) -> None:
        """Build the failure function for the automaton (required before searching)."""
        queue = []

        # Initialize root's children
        for child in self.root.children.values():
            child.fail = self.root
            queue.append(child)

        # BFS to build failure links
        while queue:
            current = queue.pop(0)

            for char, child in current.children.items():
                queue.append(child)

                # Find failure link
                fail_node = current.fail
                while fail_node and char not in fail_node.children:
                    fail_node = fail_node.fail

                child.fail = fail_node.children[char] if fail_node else self.root

                # Merge outputs
                if child.fail.output:
                    child.output.extend(child.fail.output)

    def search(self, text: str) -> List[Tuple[int, str]]:
        """
        Search for all patterns in the text.

        Returns:
            List of (position, pattern) tuples where matches were found.
        """
        matches = []
        node = self.root

        for i, char in enumerate(text):
            # Follow failure links if character not found
            while node and char not in node.children:
                node = node.fail

            if not node:
                node = self.root
                continue

            node = node.children[char]

            # Record all matches at this position
            for pattern in node.output:
                matches.append((i - len(pattern) + 1, pattern))

        return matches

    def contains_any(self, text: str) -> bool:
        """Fast check if text contains any of the patterns."""
        node = self.root

        for char in text:
            while node and char not in node.children:
                node = node.fail

            if not node:
                node = self.root
                continue

            node = node.children[char]

            if node.output:
                return True

        return False


class SecurityMatcher:
    """
    Pre-configured matchers for common security checks.
    """

    @staticmethod
    def create_deprecated_python_matcher() -> AhoCorasickMatcher:
        """Create matcher for deprecated Python functions and methods."""
        deprecated_patterns = [
            # Deprecated Python 2/3 functions
            "execfile",
            "reload",
            "unichr",
            "xrange",
            "raw_input",
            # Deprecated modules
            "imp.load_source",
            "imp.find_module",
            # Security concerns
            "eval",
            "exec",
            "compile",
            "__import__",
            # Deprecated OpenSSL
            "md5",
            "sha1",
            # Deprecated HTTP methods
            "urllib.urlopen",
            "urllib2.urlopen",
        ]

        matcher = AhoCorasickMatcher()
        matcher.add_patterns(deprecated_patterns)
        matcher.build()
        return matcher

    @staticmethod
    def create_deprecated_javascript_matcher() -> AhoCorasickMatcher:
        """Create matcher for deprecated JavaScript functions."""
        deprecated_patterns = [
            # Deprecated DOM methods
            "document.write",
            "document.writeln",
            # Deprecated jQuery methods
            ".live(",
            ".bind(",
            ".delegate(",
            # Deprecated Node.js
            "Buffer(",
            "new Buffer(",
            # Security concerns
            "eval(",
            "innerHTML",
            "outerHTML",
            "document.cookie",
        ]

        matcher = AhoCorasickMatcher()
        matcher.add_patterns(deprecated_patterns)
        matcher.build()
        return matcher

    @staticmethod
    def create_deprecated_java_matcher() -> AhoCorasickMatcher:
        """Create matcher for deprecated Java methods."""
        deprecated_patterns = [
            # Deprecated Java methods
            "Thread.stop()",
            "Thread.suspend()",
            "Thread.resume()",
            "Runtime.runFinalizersOnExit",
            # Security
            "System.getSecurityManager",
            "SecurityManager.checkMemberAccess",
            # Deprecated date/time
            "Date.getYear()",
            "Date.getMonth()",
            "Date.getDay()",
        ]

        matcher = AhoCorasickMatcher()
        matcher.add_patterns(deprecated_patterns)
        matcher.build()
        return matcher

    @staticmethod
    def create_vulnerable_package_matcher(packages: List[str]) -> AhoCorasickMatcher:
        """Create matcher for known vulnerable packages."""
        matcher = AhoCorasickMatcher()
        matcher.add_patterns(packages)
        matcher.build()
        return matcher


class HybridImportExtractor:
    """
    Hybrid approach: AST-first, with Aho-Corasick validation.

    Workflow:
    1. Try AST parsing (most accurate)
    2. On success: Validate with Aho-Corasick for security
    3. On failure: Fallback to regex
    """

    def __init__(self, forbidden_patterns: List[str] = None):
        self.forbidden_matcher = None
        if forbidden_patterns:
            self.forbidden_matcher = AhoCorasickMatcher()
            self.forbidden_matcher.add_patterns(forbidden_patterns)
            self.forbidden_matcher.build()

    def validate_security(self, code: str) -> Dict[str, List[Tuple[int, str]]]:
        """
        Fast security validation using Aho-Corasick.

        Returns:
            Dict with 'forbidden_found' containing matches.
        """
        if not self.forbidden_matcher:
            return {"forbidden_found": []}

        matches = self.forbidden_matcher.search(code)
        return {"forbidden_found": matches}


# Example: Load 10,000+ patterns from a file
class PatternLoader:
    """Load patterns from various sources for bulk validation."""

    @staticmethod
    def load_from_file(filepath: str) -> List[str]:
        """Load patterns from a newline-separated file."""
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def load_cve_patterns(language: str) -> List[str]:
        """Load known CVE-related patterns for a language (placeholder)."""
        # In production, this would fetch from a CVE database
        patterns_db = {
            "python": ["pickle.loads", "yaml.load", "marshal.loads"],
            "javascript": ["eval(", "Function(", "setTimeout("],
            "java": ["Runtime.exec", "ProcessBuilder"],
        }
        return patterns_db.get(language, [])


if __name__ == "__main__":
    # Benchmark example
    import time

    # Create matcher with 1000 patterns
    matcher = AhoCorasickMatcher()
    patterns = [f"deprecated_function_{i}" for i in range(1000)]
    matcher.add_patterns(patterns)
    matcher.build()

    # Test text
    test_code = "some code here " * 1000 + "deprecated_function_500" + " more code"

    # Measure search time
    start = time.time()
    matches = matcher.search(test_code)
    elapsed = time.time() - start

    print(f"Searched {len(test_code)} chars for {len(patterns)} patterns")
    print(f"Found {len(matches)} matches in {elapsed*1000:.2f}ms")
    print(f"Matches: {matches}")
