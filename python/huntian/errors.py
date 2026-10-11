"""Exception hierarchy for Huntian.

Every error raised by the package derives from :class:`HuntianError`, so a caller
can catch the whole family with a single ``except HuntianError``.

Validation problems are reported as structured :class:`Issue` records carrying
a dotted *path* (``device.structure.layers[1].thickness_nm``), which is what
makes failures from LLM-generated input actionable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "Issue",
    "format_issues",
    "HuntianError",
    "SchemaError",
    "DeckError",
    "VersionError",
    "RequirementError",
    "RegistryError",
    "ModuleUnavailableError",
    "VersionConflictError",
    "DependencyError",
    "DuplicateModuleError",
    "ModuleConfigError",
    "ServiceUnavailableError",
    "ModuleLifecycleError",
    "ContextError",
    "ArtifactMissingError",
    "PipelineError",
]


@dataclass(frozen=True)
class Issue:
    """A single validation problem addressed by a dotted path."""

    path: str
    message: str
    code: str = "invalid"
    hint: str | None = None

    def __str__(self) -> str:
        base = f"{self.path}: {self.message}" if self.path else self.message
        return f"{base} ({self.hint})" if self.hint else base


def format_issues(issues: Iterable[Issue], *, indent: str = "  ") -> str:
    """Render an iterable of issues as an indented bullet list."""
    rendered = [f"{indent}- {issue}" for issue in issues]
    return "\n".join(rendered) if rendered else f"{indent}(no issues)"


class HuntianError(Exception):
    """Base class for every error raised by Huntian."""


class SchemaError(HuntianError):
    """One or more fields failed structural or semantic validation."""

    def __init__(self, issues: Iterable[Issue], *, where: str | None = None) -> None:
        self.issues: tuple[Issue, ...] = tuple(issues)
        self.where = where
        header = f"schema validation failed for {where}" if where else "schema validation failed"
        super().__init__(f"{header}:\n{format_issues(self.issues)}")


class DeckError(HuntianError):
    """A Deck document could not be read, parsed or resolved."""


class VersionError(HuntianError):
    """A version string could not be parsed."""


class RequirementError(HuntianError):
    """A module requirement string could not be parsed."""


class RegistryError(HuntianError):
    """Base class for module registry failures."""


class ModuleUnavailableError(RegistryError):
    """No registered module satisfies a requirement."""


class VersionConflictError(RegistryError):
    """Two requirements ask for mutually exclusive versions of one module."""


class DependencyError(RegistryError):
    """The module dependency graph is cyclic or otherwise unresolvable."""


class DuplicateModuleError(RegistryError):
    """A module id/version pair is already registered."""


class ModuleConfigError(HuntianError):
    """Module parameters are missing, unknown or out of range."""


class ServiceUnavailableError(RegistryError):
    """A collaborator service could not be resolved for a stage."""


class ModuleLifecycleError(HuntianError):
    """A module method was called in an invalid lifecycle state."""


class ContextError(HuntianError):
    """Base class for simulation-context failures."""


class ArtifactMissingError(ContextError):
    """A required artifact is not present in the simulation context."""


class PipelineError(HuntianError):
    """A pipeline stage failed."""
