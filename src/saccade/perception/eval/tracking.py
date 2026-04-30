class GlobalTrackIdMapper:
    """Map per-sequence local track IDs into run-global unique numeric IDs."""

    def __init__(self) -> None:
        self._next_global_id = 1
        self._mapping: dict[tuple[str, int], int] = {}

    def map(self, sequence: str, local_track_id: int) -> int:
        key = (sequence, int(local_track_id))
        global_id = self._mapping.get(key)
        if global_id is None:
            global_id = self._next_global_id
            self._mapping[key] = global_id
            self._next_global_id += 1
        return global_id

    def dump_lines(self) -> list[str]:
        lines = []
        for (sequence, local_id), global_id in sorted(
            self._mapping.items(), key=lambda item: item[1]
        ):
            lines.append(f"{sequence}\tlocal_id={local_id}\tglobal_id={global_id}")
        return lines
