#!/usr/bin/env python3
"""T2 (moments at creation), T3 (same seed same vector), T4 (trait constant across episodes)."""
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ps_common import BASE_K, RESULTS, base_args, run_job  # noqa: E402
from ps_analysis import load_trace  # noqa: E402


def trait_file(name):
    return RESULTS / "t2" / name / "traits.csv"


def run(name, extra, seed, weeks, k=BASE_K):
    return run_job(f"t2/{name}", base_args(k, seed, weeks) + extra, extra_outputs=[("traits.csv", "--traitDump")])


def moments(path):
    t, w = [], []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            t.append(float(row["trait"]))
            w.append(float(row["susceptibilityWeight"]))
    n = len(t)
    mean = sum(t) / n
    var = sum((x - mean) ** 2 for x in t) / (n - 1)
    return {"n": n, "mean": mean, "cv2": var / mean ** 2, "susc_mean": sum(w) / n, "trait": t}


cases = {
    "gamma_rho0.5_s5": (["--traitShape", "0.3", "--traitSusceptibilityExp", "0.5"], 5, 1),
    "gamma_rho0_s5": (["--traitShape", "0.3", "--traitSusceptibilityExp", "0"], 5, 1),
    "gamma_rho1_s5": (["--traitShape", "0.3", "--traitSusceptibilityExp", "1"], 5, 1),
    "gamma_rho0.5_s6": (["--traitShape", "0.3", "--traitSusceptibilityExp", "0.5"], 6, 1),
    "gamma_a0.08_s5": (["--traitShape", "0.08", "--traitSusceptibilityExp", "0.25"], 5, 1),
    "lognormal_rho1_s5": (["--traitDist", "lognormal", "--traitShape", "0.3", "--traitSusceptibilityExp", "1"], 5, 1),
    "none_s5": ([], 5, 1),
    "reinfection": (["--traitShape", "0.3", "--traitSusceptibilityExp", "0"], 5, 8),
}
for name, (extra, seed, weeks) in cases.items():
    run(name, extra, seed, weeks)

report = {}
m = {n: moments(trait_file(n)) for n in cases}
for n in ("gamma_rho0.5_s5", "gamma_rho1_s5", "gamma_a0.08_s5", "lognormal_rho1_s5", "gamma_rho0_s5"):
    a = 0.08 if "a0.08" in n else 0.3
    target = 1 / a
    r = m[n]
    ok = abs(r["mean"] - 1) <= 0.01 and abs(r["cv2"] / target - 1) <= 0.05 and abs(r["susc_mean"] - 1) <= 0.01
    report[f"T2_{n}"] = {"mean": r["mean"], "cv2": r["cv2"], "cv2_target": target, "susc_mean": r["susc_mean"], "pass": ok}
t3_same = m["gamma_rho0.5_s5"]["trait"] == m["gamma_rho0_s5"]["trait"] == m["gamma_rho1_s5"]["trait"]
t3_diff = m["gamma_rho0.5_s5"]["trait"] != m["gamma_rho0.5_s6"]["trait"]
report["T3"] = {"same_seed_same_trait_across_rho": t3_same, "different_seed_differs": t3_diff, "pass": t3_same and t3_diff}
trace = load_trace(RESULTS / "t2" / "reinfection" / "trace.tsv")
trait = m["reinfection"]["trait"]
reinfected = [r for r in trace if r["count"] > 1]
bad = [r for r in trace if abs(r["shed"] - trait[r["agent"]]) > 1e-5 * max(1.0, trait[r["agent"]])]
report["T4"] = {"infected_agents": len(trace), "reinfected_agents": len(reinfected), "agents_with_shed_not_equal_trait": len(bad),
                "pass": len(reinfected) > 0 and not bad}
(RESULTS / "t2" / "result.json").write_text(json.dumps(report, indent=1) + "\n")
for key, value in report.items():
    print(key, value)
sys.exit(0 if all(v["pass"] for v in report.values()) else 1)
