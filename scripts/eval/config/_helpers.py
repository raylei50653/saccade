def _help(text: str, *, range_hint: str | None = None, edge: str | None = None) -> str:
    parts = [text]
    if range_hint:
        parts.append(f"Range: {range_hint}.")
    if edge:
        parts.append(f"Boundary: {edge}.")
    return " ".join(parts)


def _tier(name: str, level: str) -> str:
    return f"{name} [{level}]"
