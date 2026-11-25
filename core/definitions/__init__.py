"""Nạp và kiểm tra định nghĩa tri thức theo Contract v2."""

from core.definitions.loader import (
    DefinitionIssue,
    DefinitionLoadError,
    DefinitionLoader,
    LoadedDefinition,
    adapt_legacy_definition,
)
from core.definitions.registry import DefinitionRegistry, DefinitionRegistryError

__all__ = [
    "DefinitionIssue",
    "DefinitionLoadError",
    "DefinitionLoader",
    "DefinitionRegistry",
    "DefinitionRegistryError",
    "LoadedDefinition",
    "adapt_legacy_definition",
]
