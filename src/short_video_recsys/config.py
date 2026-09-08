"""Configuration loading with small, explicit validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration is missing required values."""


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and reject empty or non-mapping documents."""

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"Expected a YAML mapping in {path}")
    return config


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a required nested mapping with a useful error message."""

    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid mapping: {key}")
    return value
