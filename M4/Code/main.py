"""Single entry point for the isolated CSE516 M4 experiment suite."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT / "Data" / "survey lung cancer.csv"
RESULTS = PROJECT / "Results"

def run(command: list[str]) -> None:
    print("\n$", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=HERE, check=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="CSE516 M4 complete reproducible experiment suite")
    parser.add_argument("--experiment", choices=["all", "m3", "m4"], default="all")
    parser.add_argument("--seed", type=int, default=516)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-cond", type=int, default=3)
    parser.add_argument("--queries-per-size", type=int, default=3)
    parser.add_argument("--skip-heavy-structure", action="store_true",
                        help="Reserved for future selective runs; full run remains default.")
    args = parser.parse_args()
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    m3_out = RESULTS / "M3"
    m4_out = RESULTS / "M4" / "Statistical_Analysis"
    if args.experiment in ("all", "m3"):
        run([sys.executable, "m3_experiments.py", "--data", str(DATA),
             "--out", str(m3_out), "--seed", str(args.seed),
             "--folds", str(args.folds), "--repeats", str(args.repeats),
             "--bootstrap", str(args.bootstrap), "--alpha", str(args.alpha),
             "--max-cond", str(args.max_cond)])
    if args.experiment in ("all", "m4"):
        repeat_csv = m3_out / "repeat_level_metrics.csv"
        if not repeat_csv.exists():
            raise FileNotFoundError(
                f"{repeat_csv} is missing. Run --experiment m3 first or use all.")
        run([sys.executable, "m4_experiments.py", "--data", str(DATA),
             "--m3-results", str(m3_out), "--out", str(m4_out),
             "--seed", str(args.seed), "--alpha", str(args.alpha),
             "--max-cond", str(args.max_cond),
             "--queries-per-size", str(args.queries_per_size)])
    print("\nExperiment suite completed. Results:", RESULTS)

if __name__ == "__main__":
    main()
