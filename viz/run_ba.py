#!/usr/bin/env python3
"""One BA.1/BA.2 trial through 31 Aug 2022.

Variants 0–2 and 5–6 stay at the fitted or default multipliers. The arguments
change BA.1 (variant 3) and BA.2 (variant 4): infectiousness, progression, and
the early ExposeToMutation day and daily fraction. The later variant-4 rule
(day 1530) is left alone.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/ireguly/pansim")
sys.path.insert(0, str(ROOT / "viz"))
import score_ba

BIN = ROOT / "build_gpu" / "panSim"
CLOSURE_SRC = ROOT / "inputConfigFiles/closureJun6_real_later3_delta_omicronBA2_earlier8.json"
OUT = Path("/tmp/pansim-fit/ba")
WEEKS = 102
INF_FIXED = [0.98, 1.85, 1.98]
PROG_FIXED = [0.94, 1.03, 0.84]
INF_LAST = 6.8
PROG_LAST = 0.45


def write_closure(day3: int, frac3: float, day4: int, frac4: float, day5: int, frac5: float, dest: Path) -> None:
    rules = json.loads(CLOSURE_SRC.read_text())
    seen = set()
    for rule in rules["rules"]:
        if rule.get("name") != "ExposeToMutation":
            continue
        variant = int(rule.get("threshold2", -1))
        if variant not in (3, 4, 5) or variant in seen:
            continue
        if variant == 4 and int(rule["threshold"]) > 1000:
            continue
        if variant == 3:
            rule["threshold"] = int(day3)
            rule["parameter"] = float(frac3)
            rule["openAfter"] = 6
        elif variant == 4:
            rule["threshold"] = int(day4)
            rule["parameter"] = float(frac4)
            rule["openAfter"] = 6
        else:
            rule["threshold"] = int(day5)
            rule["parameter"] = float(frac5)
        seen.add(variant)
    if seen != {3, 4, 5}:
        raise RuntimeError(f"expected BA.1, BA.2 and variant 5 rules, found {seen}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rules, indent=2))


def run_once(inf3: float, prog3: float, inf4: float, prog4: float, inf5: float, prog5: float, dest: Path, closure: Path) -> None:
    inf = ",".join(str(x) for x in [*INF_FIXED, inf3, inf4, inf5, INF_LAST])
    prog = ",".join(str(x) for x in [*PROG_FIXED, prog3, prog4, prog5, PROG_LAST])
    cmd = [
        str(BIN),
        "-r",
        "--quarantinePolicy",
        "3",
        "-k",
        "0.00041",
        "--progression",
        "inputConfigFiles/progressions_Jun17_tune/transition_config.json",
        "-A",
        "inputConfigFiles/agentTypes_3.json",
        "-a",
        "inputRealExample/agents1.json",
        "-l",
        "inputRealExample/locations0.json",
        "--infectiousnessMultiplier",
        inf,
        "--diseaseProgressionScaling",
        prog,
        "--closures",
        str(closure),
        "-w",
        str(WEEKS),
    ]
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with dest.open("w") as stdout:
        subprocess.run(cmd, cwd=ROOT, stdout=stdout, stderr=subprocess.DEVNULL, check=True, timeout=420)
    print(f"  {dest.name} {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--inf3", type=float, default=2.58)
    parser.add_argument("--prog3", type=float, default=0.72)
    parser.add_argument("--day3", type=int, default=437)
    parser.add_argument("--frac3", type=float, default=25e-5)
    parser.add_argument("--inf4", type=float, default=4.32)
    parser.add_argument("--prog4", type=float, default=0.57)
    parser.add_argument("--day4", type=int, default=480)
    parser.add_argument("--frac4", type=float, default=20e-5)
    parser.add_argument("--inf5", type=float, default=6.8)
    parser.add_argument("--prog5", type=float, default=0.463)
    parser.add_argument("--day5", type=int, default=604)
    parser.add_argument("--frac5", type=float, default=10e-5)
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()
    folder = OUT / args.tag
    folder.mkdir(parents=True, exist_ok=True)
    meta = {
        "inf3": args.inf3,
        "prog3": args.prog3,
        "day3": args.day3,
        "frac3": args.frac3,
        "inf4": args.inf4,
        "prog4": args.prog4,
        "day4": args.day4,
        "frac4": args.frac4,
        "inf5": args.inf5,
        "prog5": args.prog5,
        "day5": args.day5,
        "frac5": args.frac5,
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    closure = folder / "closures.json"
    write_closure(args.day3, args.frac3, args.day4, args.frac4, args.day5, args.frac5, closure)
    have = sorted(folder.glob("run*.stdout"))
    while len(have) < args.n:
        dest = folder / f"run{len(have):02d}.stdout"
        run_once(args.inf3, args.prog3, args.inf4, args.prog4, args.inf5, args.prog5, dest, closure)
        have = sorted(folder.glob("run*.stdout"))
    print(
        f"tag {args.tag} inf3 {args.inf3} prog3 {args.prog3} day3 {args.day3} frac3 {args.frac3} "
        f"inf4 {args.inf4} prog4 {args.prog4} day4 {args.day4} frac4 {args.frac4} "
        f"inf5 {args.inf5} prog5 {args.prog5} day5 {args.day5} frac5 {args.frac5}",
        flush=True,
    )
    # Reuse the scorer's printer.
    sys.argv = ["score_ba.py", *[str(p) for p in have]]
    score_ba.main()


if __name__ == "__main__":
    main()
