"""Strict JSONL persistence for M0 counterfactual route records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .types import CounterfactualRecord


def append_records(path: str | Path, records: Iterable[CounterfactualRecord]) -> int:
    """Append validated records and return their count.

    The caller owns process-level coordination; each record is serialized as one
    complete JSON line so interrupted collection never creates a partial record.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def read_records(path: str | Path) -> Iterator[CounterfactualRecord]:
    """Yield schema-validated records with line-numbered parse errors."""

    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                yield CounterfactualRecord.from_dict(payload)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid record at {source}:{line_number}: {error}") from error
