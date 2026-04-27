"""
Batch experiment runner.
"""

import argparse
import itertools
import subprocess
import sys

from config import DATASET_LIST, MODEL_LIST


def parse_args():
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument("--models", nargs="+", default=MODEL_LIST, help="Models to evaluate")
    parser.add_argument("--datasets", nargs="+", default=DATASET_LIST, help="Datasets to evaluate")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["threshold", "likelihood", "classifier"],
        help="Attack strategies",
    )
    parser.add_argument("--num_queries", type=int, default=50)
    parser.add_argument("--num_members", type=int, default=500)
    parser.add_argument("--num_non_members", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt_mode", type=str, default="raw_prefix", choices=["raw_prefix", "template"])
    parser.add_argument("--calibration_mode", type=str, default="crossfit", choices=["crossfit", "same_pool"])
    parser.add_argument("--crossfit_folds", type=int, default=5)
    parser.add_argument("--polarity_mode", type=str, default="domain", choices=["auto", "domain"])
    parser.add_argument("--run_baselines", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def main():
    args = parse_args()
    combos = list(itertools.product(args.models, args.datasets, args.strategies))
    print(f"Total experiments: {len(combos)}")

    for model, dataset, strategy in combos:
        cmd = [
            sys.executable,
            "run_experiment.py",
            "--model",
            model,
            "--dataset",
            dataset,
            "--strategy",
            strategy,
            "--num_queries",
            str(args.num_queries),
            "--num_members",
            str(args.num_members),
            "--num_non_members",
            str(args.num_non_members),
            "--prompt_mode",
            args.prompt_mode,
            "--calibration_mode",
            args.calibration_mode,
            "--crossfit_folds",
            str(args.crossfit_folds),
            "--polarity_mode",
            args.polarity_mode,
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--output_dir",
            args.output_dir,
        ]
        if args.run_baselines:
            cmd.append("--run_baselines")

        print("\n" + "=" * 70)
        print(f"  Model: {model} | Dataset: {dataset} | Strategy: {strategy}")
        print("=" * 70)
        print(f"  CMD: {' '.join(cmd)}")

        if not args.dry_run:
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                print(f"  WARNING: Experiment failed with code {result.returncode}")

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()