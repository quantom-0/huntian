"""Version metadata for the NemoTCAD package."""

from __future__ import annotations

__version__ = "0.1.0"

#: Version of the *module plugin contract* (``IModule`` / ``ModuleManifest``).
#: Plugins declare the contract version they were written against; the registry
#: refuses to load a plugin whose contract major version differs.
API_VERSION = "1.0"

#: Version of the on-disk Deck document format.
SCHEMA_VERSION = "1.0"

__all__ = ["__version__", "API_VERSION", "SCHEMA_VERSION"]
