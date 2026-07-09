def _help(
    text: str,
    *,
    range_hint: str | None = None,
    edge: str | None = None,
    policy: str | None = None,
) -> str:
    """Build argparse help text.

    ``policy`` tags developer-facing decision surface (does not change defaults):
      ACTIVE     — production path knob
      PRESET-OFF — schema may default on; headline YAML forces off
      LATENT     — wired for ablation; not a safe default
      NO-GO      — evidence rejected as headline policy
      ENV        — env-only; not headline YAML
    """
    parts: list[str] = []
    if policy:
        parts.append(f"[{policy}]")
    parts.append(text)
    if range_hint:
        parts.append(f"Range: {range_hint}.")
    if edge:
        parts.append(f"Boundary: {edge}.")
    return " ".join(parts)


def _tier(name: str, level: str) -> str:
    return f"{name} [{level}]"
