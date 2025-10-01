"""PyPI version resolver for checking package compatibility."""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import aiohttp
from packaging import version
from packaging.specifiers import SpecifierSet
from packaging.requirements import Requirement

logger = logging.getLogger(__name__)


class VersionResolver:
    """Resolve package versions and check compatibility."""

    def __init__(self):
        """Initialize version resolver."""
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, Dict[str, Any]] = {}

    async def __aenter__(self):
        """Create aiohttp session."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()

    async def get_package_info(self, package_name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch package information from PyPI.

        Args:
            package_name: Name of the package

        Returns:
            Package info dict or None on error
        """
        # Check cache
        if package_name in self.cache:
            logger.debug(f"Using cached info for {package_name}")
            return self.cache[package_name]

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            url = f"https://pypi.org/pypi/{package_name}/json"
            logger.info(f"Fetching package info for {package_name} from PyPI...")

            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 404:
                    logger.warning(f"Package {package_name} not found on PyPI")
                    return None

                if response.status != 200:
                    logger.error(f"PyPI API error: {response.status}")
                    return None

                data = await response.json()

                # Extract useful info
                info = {
                    "name": data["info"]["name"],
                    "version": data["info"]["version"],
                    "summary": data["info"]["summary"],
                    "requires_python": data["info"].get("requires_python"),
                    "requires_dist": data["info"].get("requires_dist", []),
                    "all_versions": list(data["releases"].keys())
                }

                # Cache it
                self.cache[package_name] = info
                return info

        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching info for {package_name}")
            return None
        except Exception as e:
            logger.error(f"Error fetching package info: {e}")
            return None

    async def get_latest_version(self, package_name: str) -> Optional[str]:
        """
        Get the latest stable version of a package.

        Args:
            package_name: Name of the package

        Returns:
            Latest version string or None
        """
        info = await self.get_package_info(package_name)
        if info:
            return info["version"]
        return None

    async def check_version_compatibility(
        self,
        package_name: str,
        python_version: str = "3.11"
    ) -> Dict[str, Any]:
        """
        Check if package is compatible with Python version.

        Args:
            package_name: Name of the package
            python_version: Python version to check (e.g., "3.11")

        Returns:
            Compatibility info dict
        """
        info = await self.get_package_info(package_name)

        if not info:
            return {
                "package": package_name,
                "compatible": False,
                "reason": "Package not found on PyPI"
            }

        requires_python = info.get("requires_python")

        if not requires_python:
            # No Python version requirement specified
            return {
                "package": package_name,
                "version": info["version"],
                "compatible": True,
                "reason": "No Python version requirement specified"
            }

        try:
            # Parse Python version requirement
            specifier = SpecifierSet(requires_python)
            compatible = python_version in specifier

            return {
                "package": package_name,
                "version": info["version"],
                "compatible": compatible,
                "requires_python": requires_python,
                "reason": f"Package requires Python {requires_python}"
            }

        except Exception as e:
            logger.error(f"Error checking Python compatibility: {e}")
            return {
                "package": package_name,
                "compatible": False,
                "reason": f"Error parsing version requirement: {str(e)}"
            }

    async def resolve_dependencies(
        self,
        packages: List[str],
        python_version: str = "3.11"
    ) -> Dict[str, Any]:
        """
        Resolve dependencies for multiple packages.

        Args:
            packages: List of package names
            python_version: Python version to target

        Returns:
            Resolution result with compatibility info
        """
        results = {
            "packages": {},
            "conflicts": [],
            "all_compatible": True
        }

        # Fetch info for all packages
        tasks = [self.check_version_compatibility(pkg, python_version) for pkg in packages]
        compatibility_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for pkg_name, compat_result in zip(packages, compatibility_results):
            if isinstance(compat_result, Exception):
                logger.error(f"Error checking {pkg_name}: {compat_result}")
                results["all_compatible"] = False
                results["packages"][pkg_name] = {
                    "compatible": False,
                    "reason": str(compat_result)
                }
                continue

            results["packages"][pkg_name] = compat_result

            if not compat_result.get("compatible", False):
                results["all_compatible"] = False

        # Check for dependency conflicts (simplified)
        await self._check_dependency_conflicts(packages, results)

        return results

    async def _check_dependency_conflicts(
        self,
        packages: List[str],
        results: Dict[str, Any]
    ):
        """
        Check for conflicts between package dependencies.

        Args:
            packages: List of package names
            results: Results dict to update with conflicts
        """
        # This is a simplified conflict checker
        # A full implementation would need to build a dependency graph

        dependency_map: Dict[str, List[str]] = {}

        for pkg_name in packages:
            info = await self.get_package_info(pkg_name)
            if not info or not info.get("requires_dist"):
                continue

            deps = []
            for req_str in info["requires_dist"]:
                if not req_str:
                    continue

                try:
                    # Parse requirement
                    req = Requirement(req_str)
                    deps.append({
                        "name": req.name,
                        "specifier": str(req.specifier) if req.specifier else "any"
                    })
                except Exception as e:
                    logger.debug(f"Could not parse requirement '{req_str}': {e}")
                    continue

            dependency_map[pkg_name] = deps

        # Check for version conflicts
        shared_deps: Dict[str, List[Dict[str, Any]]] = {}

        for pkg_name, deps in dependency_map.items():
            for dep in deps:
                dep_name = dep["name"]
                if dep_name not in shared_deps:
                    shared_deps[dep_name] = []

                shared_deps[dep_name].append({
                    "required_by": pkg_name,
                    "specifier": dep["specifier"]
                })

        # Find conflicts
        for dep_name, requirements in shared_deps.items():
            if len(requirements) > 1:
                # Multiple packages require this dependency
                specifiers = [req["specifier"] for req in requirements if req["specifier"] != "any"]

                if len(set(specifiers)) > 1:
                    # Different version requirements
                    results["conflicts"].append({
                        "dependency": dep_name,
                        "requirements": requirements,
                        "message": f"Version conflict for {dep_name}: required by multiple packages with different version constraints"
                    })

    async def suggest_compatible_versions(
        self,
        package_name: str,
        python_version: str = "3.11"
    ) -> List[str]:
        """
        Suggest compatible versions for a package.

        Args:
            package_name: Name of the package
            python_version: Python version to target

        Returns:
            List of compatible version strings
        """
        info = await self.get_package_info(package_name)

        if not info:
            return []

        requires_python = info.get("requires_python")

        if not requires_python:
            # Return latest version
            return [info["version"]]

        try:
            specifier = SpecifierSet(requires_python)

            # If current Python version is compatible, return latest
            if python_version in specifier:
                return [info["version"]]

            # Otherwise, try to find compatible versions
            compatible_versions = []
            all_versions = info.get("all_versions", [])

            # Sort versions (newest first)
            try:
                sorted_versions = sorted(
                    all_versions,
                    key=lambda v: version.parse(v),
                    reverse=True
                )
            except Exception:
                sorted_versions = all_versions

            # Check each version (limit to 10 for performance)
            for ver in sorted_versions[:10]:
                # Fetch version-specific info
                ver_info = await self._get_version_info(package_name, ver)
                if ver_info and ver_info.get("requires_python"):
                    ver_specifier = SpecifierSet(ver_info["requires_python"])
                    if python_version in ver_specifier:
                        compatible_versions.append(ver)

                if len(compatible_versions) >= 5:
                    break

            return compatible_versions

        except Exception as e:
            logger.error(f"Error suggesting versions: {e}")
            return []

    async def _get_version_info(
        self,
        package_name: str,
        package_version: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get info for a specific package version.

        Args:
            package_name: Name of the package
            package_version: Specific version

        Returns:
            Version info dict or None
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            url = f"https://pypi.org/pypi/{package_name}/{package_version}/json"

            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    return None

                data = await response.json()
                return {
                    "version": data["info"]["version"],
                    "requires_python": data["info"].get("requires_python"),
                    "requires_dist": data["info"].get("requires_dist", [])
                }

        except Exception as e:
            logger.debug(f"Error fetching version info: {e}")
            return None
