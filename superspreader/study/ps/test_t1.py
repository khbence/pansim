#!/usr/bin/env python3
"""T1: new flags at default reproduce the previous binary exactly (daily table and trace columns)."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ps_common import BASE_K, RESULTS, ROOT, base_args, run_job  # noqa: E402

OLD = ROOT / "build_gpu" / "panSim_phase3"
NEW = ROOT / "build_gpu" / "panSim"
CELLS = {"none": ["--sheddingShape", "0"], "shed0.3": ["--sheddingShape", "0.3"]}


def digest(path, columns=None):
    h = hashlib.sha256()
    with open(path) as handle:
        for line in handle:
            if "seconds" in line:
                continue
            if columns is not None:
                line = "\t".join(line.rstrip("\n").split("\t")[:columns]) + "\n"
            h.update(line.encode())
    return h.hexdigest()


results = {}
for label, extra in CELLS.items():
    args = base_args(BASE_K, 7, 6) + extra
    run_job(f"t1/old_{label}", args, binary=OLD)
    run_job(f"t1/new_{label}", args, binary=NEW)
    o, n = RESULTS / f"t1/old_{label}", RESULTS / f"t1/new_{label}"
    same_daily = digest(o / "stdout.txt") == digest(n / "stdout.txt")
    same_trace = digest(o / "trace.tsv") == digest(n / "trace.tsv", columns=13)
    results[label] = {"daily_identical": same_daily, "trace_first13_identical": same_trace}
    print(label, results[label])
(RESULTS / "t1/result.json").write_text(json.dumps(results, indent=1) + "\n")
sys.exit(0 if all(all(v.values()) for v in results.values()) else 1)
