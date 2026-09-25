"""Parse panSim daily statistics from console output or a saved file.

A statistics line is tab-separated. The fields named MUT, IMM and INFV contain
comma-separated per-variant numbers and are expanded to MUT1.., IMM1.., INFV1...
Any other console text (timing report, warnings) is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .columns import EXPANDABLE, STOCK


@dataclass
class Run:
    path: str
    columns: list[str]
    data: dict[str, np.ndarray]
    n_days: int

    def population(self) -> float:
        present = [c for c in STOCK if c in self.data]
        if not present:
            return float("nan")
        total = np.zeros(self.n_days)
        for c in present:
            total += np.nan_to_num(self.data[c])
        # The partition is stable; use the median so a truncated last line cannot dominate.
        return float(np.median(total))


@dataclass
class RunSet:
    name: str
    runs: list[Run] = field(default_factory=list)

    def population(self) -> float:
        pops = [r.population() for r in self.runs if np.isfinite(r.population())]
        if not pops:
            return float("nan")
        return float(np.median(pops))


def _expand_header(header_fields: list[str], sample_fields: list[str] | None) -> list[str]:
    names: list[str] = []
    for i, name in enumerate(header_fields):
        if name in EXPANDABLE:
            width = 1
            if sample_fields is not None and i < len(sample_fields) and "," in sample_fields[i]:
                width = sample_fields[i].count(",") + 1
            for k in range(1, width + 1):
                names.append(f"{name}{k}")
        else:
            names.append(name)
    return names


def _expand_row(header_fields: list[str], fields: list[str]) -> list[float] | None:
    if len(fields) != len(header_fields):
        return None
    values: list[float] = []
    for name, raw in zip(header_fields, fields):
        raw = raw.strip()
        if name in EXPANDABLE:
            parts = raw.split(",") if raw else ["nan"]
            for part in parts:
                try:
                    values.append(float(part))
                except ValueError:
                    return None
        else:
            if "," in raw:
                return None
            try:
                values.append(float(raw))
            except ValueError:
                return None
    return values


def parse_text(text: str, source: str = "<memory>") -> Run | None:
    header: list[str] | None = None
    rows: list[list[float]] = []
    names: list[str] | None = None
    for line in text.splitlines():
        if "\t" not in line:
            continue
        fields = [p.strip() for p in line.rstrip("\n").split("\t")]
        if header is None:
            if "S" in fields and "NI" in fields:
                header = fields
            continue
        parsed = _expand_row(header, fields)
        if parsed is None:
            continue
        if names is None:
            names = _expand_header(header, fields)
        if len(parsed) != len(names):
            # A later day printed a different variant width; keep the run consistent.
            continue
        rows.append(parsed)
    if not names or not rows:
        return None
    array = np.asarray(rows, dtype=float)
    data = {name: array[:, i] for i, name in enumerate(names)}
    return Run(path=source, columns=names, data=data, n_days=array.shape[0])


def parse_file(path: str | Path) -> Run | None:
    path = Path(path)
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    return parse_text(text, source=str(path))


def load_run_set(name: str, paths: list[str | Path]) -> RunSet:
    runs: list[Run] = []
    for path in paths:
        run = parse_file(path)
        if run is not None and run.n_days > 0:
            runs.append(run)
    return RunSet(name=name, runs=runs)


def collect_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    files: list[Path] = []
    for pattern in ("*.tsv", "*.txt", "*.out", "*.stdout", "*.log"):
        files.extend(sorted(path.glob(pattern)))
    # Direct children only. Nested scenario folders are separate groups.
    return files
