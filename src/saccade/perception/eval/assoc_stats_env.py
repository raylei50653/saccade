"""Fail-closed parser for SACCADE_ASSOC_STATS.

Keep tokens in sync with include/saccade/env_flag.hpp.
"""

from __future__ import annotations

import os

FALSE_TOKENS = frozenset({"0", "false", "no", "off"})
TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})


def assoc_stats_env_enabled(value: str | None = None) -> bool:
    raw = os.environ.get("SACCADE_ASSOC_STATS", "") if value is None else value
    token = str(raw).strip().lower()
    if not token or token in FALSE_TOKENS:
        return False
    return token in TRUE_TOKENS
