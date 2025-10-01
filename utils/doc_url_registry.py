"""Registry of known documentation URLs for popular libraries.

This eliminates the need for web search API calls.
"""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Registry of known documentation URLs
DOCUMENTATION_REGISTRY: Dict[str, Dict[str, str]] = {
    "fastmcp": {
        "base_url": "https://pypi.org/project/fastmcp/",
        "docs_url": "https://pypi.org/project/fastmcp/",
        "type": "pypi"
    },
    "requests": {
        "base_url": "https://requests.readthedocs.io",
        "docs_url": "https://requests.readthedocs.io/en/latest/",
        "type": "readthedocs"
    },
    "fastapi": {
        "base_url": "https://fastapi.tiangolo.com",
        "docs_url": "https://fastapi.tiangolo.com/",
        "type": "custom_docs"
    },
    "django": {
        "base_url": "https://docs.djangoproject.com",
        "docs_url": "https://docs.djangoproject.com/en/stable/",
        "type": "custom_docs"
    },
    "flask": {
        "base_url": "https://flask.palletsprojects.com",
        "docs_url": "https://flask.palletsprojects.com/en/latest/",
        "type": "custom_docs"
    },
    "numpy": {
        "base_url": "https://numpy.org/doc",
        "docs_url": "https://numpy.org/doc/stable/",
        "type": "custom_docs"
    },
    "pandas": {
        "base_url": "https://pandas.pydata.org/docs",
        "docs_url": "https://pandas.pydata.org/docs/",
        "type": "custom_docs"
    },
    "pytorch": {
        "base_url": "https://pytorch.org/docs",
        "docs_url": "https://pytorch.org/docs/stable/",
        "type": "custom_docs"
    },
    "tensorflow": {
        "base_url": "https://www.tensorflow.org/api_docs",
        "docs_url": "https://www.tensorflow.org/api_docs/python/tf",
        "type": "custom_docs"
    },
    "langchain": {
        "base_url": "https://python.langchain.com",
        "docs_url": "https://python.langchain.com/docs/",
        "type": "custom_docs"
    },
    "pydantic": {
        "base_url": "https://docs.pydantic.dev",
        "docs_url": "https://docs.pydantic.dev/latest/",
        "type": "custom_docs"
    },
    "sqlalchemy": {
        "base_url": "https://docs.sqlalchemy.org",
        "docs_url": "https://docs.sqlalchemy.org/en/latest/",
        "type": "custom_docs"
    },
    "celery": {
        "base_url": "https://docs.celeryq.dev",
        "docs_url": "https://docs.celeryq.dev/en/stable/",
        "type": "custom_docs"
    },
    "pytest": {
        "base_url": "https://docs.pytest.org",
        "docs_url": "https://docs.pytest.org/en/latest/",
        "type": "custom_docs"
    },
    "redis": {
        "base_url": "https://redis-py.readthedocs.io",
        "docs_url": "https://redis-py.readthedocs.io/en/stable/",
        "type": "readthedocs"
    },
    "boto3": {
        "base_url": "https://boto3.amazonaws.com/v1/documentation/api/latest",
        "docs_url": "https://boto3.amazonaws.com/v1/documentation/api/latest/index.html",
        "type": "custom_docs"
    },
    "aiohttp": {
        "base_url": "https://docs.aiohttp.org",
        "docs_url": "https://docs.aiohttp.org/en/stable/",
        "type": "custom_docs"
    },
    "httpx": {
        "base_url": "https://www.python-httpx.org",
        "docs_url": "https://www.python-httpx.org/",
        "type": "custom_docs"
    },
}


def get_documentation_urls(library_name: str, max_urls: int = 10) -> List[str]:
    """
    Get documentation URLs for a library from the registry.

    Args:
        library_name: Name of the library
        max_urls: Maximum number of URLs to return

    Returns:
        List of documentation URLs
    """
    library_name = library_name.lower().strip()

    if library_name in DOCUMENTATION_REGISTRY:
        info = DOCUMENTATION_REGISTRY[library_name]
        logger.info(f"Found {library_name} in registry: {info['docs_url']}")

        # Return base URL as starting point for crawling
        return [info['docs_url']]

    # Fallback: try common documentation URL patterns
    return _try_common_patterns(library_name, max_urls)


def _try_common_patterns(library_name: str, max_urls: int) -> List[str]:
    """Try common documentation URL patterns."""
    urls = []

    # Pattern 1: ReadTheDocs
    urls.append(f"https://{library_name}.readthedocs.io/en/latest/")
    urls.append(f"https://{library_name}.readthedocs.io/en/stable/")

    # Pattern 2: GitHub Pages
    urls.append(f"https://{library_name}.github.io/")

    # Pattern 3: Custom domain
    urls.append(f"https://{library_name}.org/docs/")
    urls.append(f"https://docs.{library_name}.org/")
    urls.append(f"https://www.{library_name}.org/documentation/")

    # Pattern 4: Python package docs
    urls.append(f"https://pypi.org/project/{library_name}/")

    logger.info(f"Trying common URL patterns for {library_name}")
    return urls[:max_urls]


def add_library_to_registry(
    library_name: str,
    docs_url: str,
    base_url: Optional[str] = None,
    doc_type: str = "custom_docs"
) -> None:
    """
    Add a library to the registry.

    Args:
        library_name: Name of the library
        docs_url: Main documentation URL
        base_url: Base URL (optional, defaults to docs_url)
        doc_type: Type of documentation (github_readme, readthedocs, custom_docs)
    """
    DOCUMENTATION_REGISTRY[library_name.lower()] = {
        "base_url": base_url or docs_url,
        "docs_url": docs_url,
        "type": doc_type
    }
    logger.info(f"Added {library_name} to documentation registry")


def list_registered_libraries() -> List[str]:
    """Get list of all registered libraries."""
    return sorted(DOCUMENTATION_REGISTRY.keys())


# Expand registry with common patterns for popular libraries
def expand_registry_with_pypi_top_packages():
    """Add top PyPI packages to registry using common patterns."""
    top_packages = [
        # Web frameworks
        ("starlette", "https://www.starlette.io/"),
        ("uvicorn", "https://www.uvicorn.org/"),
        ("sanic", "https://sanic.dev/"),

        # Data science
        ("scikit-learn", "https://scikit-learn.org/stable/"),
        ("matplotlib", "https://matplotlib.org/stable/"),
        ("scipy", "https://docs.scipy.org/doc/scipy/"),
        ("seaborn", "https://seaborn.pydata.org/"),

        # Database
        ("psycopg2", "https://www.psycopg.org/docs/"),
        ("pymongo", "https://pymongo.readthedocs.io/en/stable/"),
        ("elasticsearch", "https://elasticsearch-py.readthedocs.io/en/stable/"),

        # Async
        ("asyncio", "https://docs.python.org/3/library/asyncio.html"),
        ("trio", "https://trio.readthedocs.io/en/stable/"),

        # Testing
        ("unittest", "https://docs.python.org/3/library/unittest.html"),
        ("mock", "https://docs.python.org/3/library/unittest.mock.html"),
        ("tox", "https://tox.wiki/en/latest/"),

        # CLI
        ("click", "https://click.palletsprojects.com/"),
        ("typer", "https://typer.tiangolo.com/"),
        ("argparse", "https://docs.python.org/3/library/argparse.html"),

        # Utils
        ("dotenv", "https://pypi.org/project/python-dotenv/"),
        ("pyyaml", "https://pyyaml.org/wiki/PyYAMLDocumentation"),
        ("pillow", "https://pillow.readthedocs.io/en/stable/"),
    ]

    for lib_name, url in top_packages:
        if lib_name not in DOCUMENTATION_REGISTRY:
            DOCUMENTATION_REGISTRY[lib_name] = {
                "base_url": url,
                "docs_url": url,
                "type": "custom_docs"
            }

# Auto-expand registry
expand_registry_with_pypi_top_packages()
