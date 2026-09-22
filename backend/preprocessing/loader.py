from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator


@dataclass(slots=True)
class LoadedCandidate:
    line_number: int
    raw: Dict[str, Any]


def load_candidates(path: str | Path) -> Iterator[LoadedCandidate]:
    """
    Stream candidate JSONL rows with line numbers.

    Blank lines are ignored. Invalid JSON raises a ValueError that includes
    the source line number so stage-0 can surface precise diagnostics.
    """

    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {jsonl_path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object at {jsonl_path}:{line_number}")
            yield LoadedCandidate(line_number=line_number, raw=payload)

