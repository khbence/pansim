#!/usr/bin/env python3
"""Regression: reproduce the 09 early-k cells (no multiplier 5.25 [4.35, 6.99]; shape 0.3 0.33 [0.29, 0.37]).

Same calibrated coefficients, seeds 1001..1004 and protocol as 09 section 3, run with the new binary.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ps_common import RESULTS, run_job  # noqa: E402
from ps_analysis import early_window_summary, json_ready  # noqa: E402
from run_phase2 import mech_args, open_epidemic_args  # noqa: E402

CELLS = {
    "baseline": (0.0, 0.00041, (5.25, 4.35, 6.99)),
    "shape0.3": (0.3, 0.0004836606939759114, (0.33, 0.29, 0.37)),
}
out = {}
for label, (shed, k, (target, lo, hi)) in CELLS.items():
    for rep in range(1, 5):
        run_job(f"regression/{label}/rep{rep:02d}", open_epidemic_args(mech_args(shed, 0, "off", 1, 1.0))(k, 1000 + rep, 8))
    s = early_window_summary(RESULTS / "regression" / label)
    ci = s["k_infector_ci"]
    inside = lo <= s["k"] <= hi
    out[label] = {"published": [target, lo, hi], "k": s["k"], "infectors": s["infectors"], "k_infector_ci": ci,
                  "k_cluster_ci": s["k_cluster_ci"], "inside_published_interval": inside}
    print(label, "k", round(s["k"], 3), "published", (target, lo, hi), "inside", inside, "infectors", s["infectors"])
(RESULTS / "regression" / "result.json").write_text(json.dumps(json_ready(out), indent=1) + "\n")
sys.exit(0 if all(v["inside_published_interval"] for v in out.values()) else 1)
