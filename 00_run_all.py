"""
RNQI MASTER PIPELINE RUNNER + REFERENCE STATISTICS
--------------------------------------------------
Runs the completed RNQI pipeline in order:

    02. rnqi_data.py
    03. rnqi_connectivity.py
    04. rnqi_efficiency.py
    05. rnqi_hierarchy.py
    06. rnqi_final.py

After the final composite is produced, this script calculates the reference
statistics used for interpretation from the five final RNQI city scores:

    Mean (population)
    Population SD
    Mean + 1 SD
    Mean - 1 SD

The RNQI documentation uses population SD for the structural reference
bounds, so ddof=0 is used for the primary thresholds.

It also assigns the same three-tier structure described in the documentation,
using the thresholds calculated from the five benchmark cities rather than
illustrative/sample thresholds.

IMPORTANT:
- This script does not run the hierarchy sensitivity test.
- Running this master script reruns the underlying metric calculations.
- Efficiency and hierarchy can therefore take substantial time.
- Run this script from the project root.

Usage:
    python 00_run_all.py

To run only a selected stage onward:
    python 00_run_all.py --from-step 4

Steps:
    2 = rnqi_data.py
    3 = rnqi_connectivity.py
    4 = rnqi_efficiency.py
    5 = rnqi_hierarchy.py
    6 = rnqi_final.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
METADATA = ROOT / "outputs" / "metadata"

PIPELINE = {
    2: "02_rnqi_data.py",
    3: "03_rnqi_connectivity.py",
    4: "04_rnqi_efficiency.py",
    5: "05_rnqi_hierarchy.py",
    6: "06_rnqi_final.py",
}


def run_stage(step: int, filename: str) -> None:
    print()
    print("=" * 76)
    print(f"STEP {step}: {filename}")
    print("=" * 76)

    script = ROOT / filename
    if not script.exists():
        raise FileNotFoundError(f"Pipeline script not found: {script}")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{filename} failed with exit code {result.returncode}. "
            "Pipeline stopped; later steps were not executed."
        )

    print(f"\nSTEP {step} COMPLETED: {filename}")


def locate_rnqi_file() -> Path:
    candidates = [
        METADATA / "final_rnqi_scores.csv",
        METADATA / "city_wise_final_rnqi_scores.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find the final RNQI score CSV. Expected one of:\n"
        + "\n".join(str(p) for p in candidates)
    )


def calculate_reference_statistics(rnqi_path: Path) -> None:
    print()
    print("=" * 76)
    print("FINAL RNQI REFERENCE STATISTICS")
    print("=" * 76)

    df = pd.read_csv(rnqi_path)

    city_col = next(
        (c for c in ["city", "City", "CITY"] if c in df.columns),
        None,
    )
    score_col = next(
        (
            c
            for c in [
                "rnqi_score",
                "final_rnqi",
                "RNQI",
                "Final RNQI Score",
            ]
            if c in df.columns
        ),
        None,
    )

    if city_col is None or score_col is None:
        raise ValueError(
            f"Could not identify city/RNQI columns in {rnqi_path}.\n"
            f"Columns found: {list(df.columns)}"
        )

    x = df[[city_col, score_col]].copy()
    x.columns = ["city", "rnqi_score"]
    x["city"] = x["city"].astype(str).str.strip()
    x["rnqi_score"] = pd.to_numeric(x["rnqi_score"], errors="coerce")

    if x["rnqi_score"].isna().any():
        raise ValueError("Final RNQI file contains missing/non-numeric scores.")

    if len(x) != 5:
        raise ValueError(
            f"Expected 5 benchmark cities, found {len(x)}."
        )

    mean = x["rnqi_score"].mean()
    population_sd = x["rnqi_score"].std(ddof=0)
    sample_sd = x["rnqi_score"].std(ddof=1)

    upper = mean + population_sd
    lower = mean - population_sd

    def tier(score: float) -> str:
        if score > upper:
            return "Tier 1: High-Performance Network"
        elif score >= lower:
            return "Tier 2: Transitional/Moderate Network"
        return "Tier 3: Critical/Low-Performance Network"

    x["reference_tier"] = x["rnqi_score"].apply(tier)

    stats = pd.DataFrame(
        [
            {
                "reference_city_count": len(x),
                "rnqi_mean": mean,
                "population_sd": population_sd,
                "sample_sd": sample_sd,
                "mean_plus_1_sd": upper,
                "mean_minus_1_sd": lower,
                "threshold_basis": "Five final benchmark RNQI scores; population SD (ddof=0)",
            }
        ]
    )

    tier_path = METADATA / "final_rnqi_reference_classification.csv"
    stats_path = METADATA / "final_rnqi_reference_statistics.csv"

    x.to_csv(tier_path, index=False)
    stats.to_csv(stats_path, index=False)

    print(f"Reference cities : {len(x)}")
    print(f"Mean (μ)         : {mean:.6f}")
    print(f"Population SD (σ): {population_sd:.6f}")
    print(f"Sample SD (s)    : {sample_sd:.6f}")
    print(f"Mean + 1 SD      : {upper:.6f}")
    print(f"Mean - 1 SD      : {lower:.6f}")

    print()
    print("CITY INTERPRETATION")
    print("-" * 76)
    print(
        x[["city", "rnqi_score", "reference_tier"]].to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}",
        )
    )

    print()
    print("Saved:")
    print(f"  {stats_path}")
    print(f"  {tier_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the RNQI pipeline and calculate final reference statistics."
    )
    parser.add_argument(
        "--from-step",
        type=int,
        choices=[2, 3, 4, 5, 6],
        default=2,
        help="Start from this pipeline step (default: 2).",
    )
    args = parser.parse_args()

    print("=" * 76)
    print("RNQI MASTER PIPELINE")
    print("=" * 76)
    print(f"Project root: {ROOT}")
    print(f"Starting from step: {args.from_step}")

    for step in range(args.from_step, 7):
        run_stage(step, PIPELINE[step])

    rnqi_path = locate_rnqi_file()
    calculate_reference_statistics(rnqi_path)

    print()
    print("=" * 76)
    print("RNQI PIPELINE COMPLETE")
    print("=" * 76)
    print("All requested stages completed successfully.")
    print("Final RNQI and reference interpretation statistics are available in:")
    print("  outputs/metadata/")


if __name__ == "__main__":
    main()
