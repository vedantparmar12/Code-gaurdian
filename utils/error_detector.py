"""Error detection for Python code using AST and static analysis."""

import ast
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ErrorDetector:
    """Detect errors in Python code using multiple analysis methods."""

    def __init__(self):
        """Initialize error detector."""
        self.errors: List[Dict[str, Any]] = []

    def detect_all_errors(self, code: str) -> List[Dict[str, Any]]:
        """
        Run all error detection methods.

        Args:
            code: Python code to analyze

        Returns:
            List of error dicts with type, message, line, severity
        """
        self.errors = []

        # 1. Check syntax errors
        syntax_errors = self._check_syntax(code)
        self.errors.extend(syntax_errors)

        # If syntax is broken, stop here (can't do further analysis)
        if syntax_errors:
            return self.errors

        # 2. Check import errors
        import_errors = self._check_imports(code)
        self.errors.extend(import_errors)

        # 3. Check undefined names
        undefined_errors = self._check_undefined_names(code)
        self.errors.extend(undefined_errors)

        # 4. Check common mistakes
        common_errors = self._check_common_mistakes(code)
        self.errors.extend(common_errors)

        return self.errors

    def _check_syntax(self, code: str) -> List[Dict[str, Any]]:
        """Check for syntax errors."""
        errors = []

        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append({
                "type": "SyntaxError",
                "message": str(e.msg),
                "line": e.lineno or 0,
                "column": e.offset or 0,
                "severity": "error",
                "fixable": True
            })
        except Exception as e:
            errors.append({
                "type": "ParseError",
                "message": str(e),
                "line": 0,
                "column": 0,
                "severity": "error",
                "fixable": True
            })

        return errors

    def _check_imports(self, code: str) -> List[Dict[str, Any]]:
        """Check for import-related errors."""
        errors = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Check if module exists
                        if not self._can_import(alias.name):
                            errors.append({
                                "type": "ImportError",
                                "message": f"Cannot import module '{alias.name}'",
                                "line": node.lineno,
                                "column": node.col_offset,
                                "severity": "error",
                                "fixable": True,
                                "module_name": alias.name
                            })

                elif isinstance(node, ast.ImportFrom):
                    if node.module and not self._can_import(node.module):
                        errors.append({
                            "type": "ImportError",
                            "message": f"Cannot import from module '{node.module}'",
                            "line": node.lineno,
                            "column": node.col_offset,
                            "severity": "error",
                            "fixable": True,
                            "module_name": node.module
                        })

        except Exception as e:
            logger.error(f"Error checking imports: {e}")

        return errors

    def _can_import(self, module_name: str) -> bool:
        """
        Check if a module can be imported.

        Args:
            module_name: Name of module to check

        Returns:
            True if module can be imported
        """
        # Get base module name
        base_module = module_name.split('.')[0]

        # Try to import
        try:
            __import__(base_module)
            return True
        except ImportError:
            return False
        except Exception:
            # If it fails for another reason, assume it exists
            return True

    def _check_undefined_names(self, code: str) -> List[Dict[str, Any]]:
        """Check for undefined variable/function names."""
        errors = []

        try:
            tree = ast.parse(code)

            # Track defined names
            defined_names = set()
            imported_names = set()

            # First pass: collect all definitions and imports
            for node in ast.walk(tree):
                # Function definitions
                if isinstance(node, ast.FunctionDef):
                    defined_names.add(node.name)

                # Class definitions
                elif isinstance(node, ast.ClassDef):
                    defined_names.add(node.name)

                # Assignments
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defined_names.add(target.id)

                # Imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_names.add(alias.asname or alias.name)

                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imported_names.add(alias.asname or alias.name)

            # Second pass: check usage
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    name = node.id

                    # Skip built-ins
                    if name in dir(__builtins__):
                        continue

                    # Check if defined or imported
                    if name not in defined_names and name not in imported_names:
                        errors.append({
                            "type": "NameError",
                            "message": f"Name '{name}' is not defined",
                            "line": node.lineno,
                            "column": node.col_offset,
                            "severity": "error",
                            "fixable": True,
                            "name": name
                        })

        except Exception as e:
            logger.error(f"Error checking undefined names: {e}")

        return errors

    def _check_common_mistakes(self, code: str) -> List[Dict[str, Any]]:
        """Check for common Python mistakes."""
        errors = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # Check for mutable default arguments
                if isinstance(node, ast.FunctionDef):
                    for i, default in enumerate(node.args.defaults):
                        if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                            arg_name = node.args.args[-(len(node.args.defaults) - i)].arg
                            errors.append({
                                "type": "MutableDefaultArgument",
                                "message": f"Mutable default argument '{arg_name}' in function '{node.name}'",
                                "line": node.lineno,
                                "column": node.col_offset,
                                "severity": "warning",
                                "fixable": True
                            })

                # Check for bare except
                if isinstance(node, ast.ExceptHandler):
                    if node.type is None:
                        errors.append({
                            "type": "BareExcept",
                            "message": "Bare 'except:' catches all exceptions including system exit",
                            "line": node.lineno,
                            "column": node.col_offset,
                            "severity": "warning",
                            "fixable": True
                        })

        except Exception as e:
            logger.error(f"Error checking common mistakes: {e}")

        return errors

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(err["severity"] == "error" for err in self.errors)

    def get_error_summary(self) -> str:
        """
        Get human-readable error summary.

        Returns:
            Formatted error summary
        """
        if not self.errors:
            return "No errors found"

        lines = ["Found the following issues:"]
        for i, error in enumerate(self.errors, 1):
            severity = error['severity'].upper()
            line_info = f"Line {error['line']}" if error['line'] > 0 else "Unknown line"
            lines.append(
                f"{i}. [{severity}] {error['type']}: {error['message']} ({line_info})"
            )

        return "\n".join(lines)

    def get_fixable_errors(self) -> List[Dict[str, Any]]:
        """Get only fixable errors."""
        return [err for err in self.errors if err.get("fixable", False)]
