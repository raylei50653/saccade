"""Strict YAML loading shared by documentation lifecycle tooling.

PyYAML's default safe loader silently keeps the last value for duplicate map
keys.  Schema validators receive only that collapsed mapping, so duplicate-key
rejection must happen while the YAML node tree is constructed.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import yaml
from yaml.nodes import MappingNode, Node


class StrictYamlError(ValueError):
    """A YAML loading failure with a stable machine-facing error class."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


class DuplicateYamlKeyError(StrictYamlError):
    """Raised before a duplicate mapping key can be silently overwritten."""

    def __init__(self, key: object, *, line: int, column: int) -> None:
        super().__init__(
            "duplicate_yaml_key",
            f"duplicate YAML key {key!r} at line {line}, column {column}",
        )


class StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader variant which rejects duplicate keys at every mapping depth."""

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        seen_keys: set[Hashable] = set()
        for key_node, _ in node.value:
            if _is_merge_key(key_node):
                key: object = "<<"
            else:
                key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                # SafeLoader will report the normal unhashable-key error.
                continue
            if key in seen_keys:
                mark = key_node.start_mark
                raise DuplicateYamlKeyError(
                    key,
                    line=mark.line + 1,
                    column=mark.column + 1,
                )
            seen_keys.add(key)
        return super().construct_mapping(node, deep=deep)


def _is_merge_key(node: Node) -> bool:
    return node.tag == "tag:yaml.org,2002:merge"


def strict_safe_load(source: str) -> Any:
    """Load YAML safely, rejecting duplicate map keys in every nested mapping."""

    try:
        return yaml.load(source, Loader=StrictSafeLoader)
    except DuplicateYamlKeyError:
        raise
    except yaml.YAMLError as error:
        raise StrictYamlError("invalid_yaml", str(error)) from error
