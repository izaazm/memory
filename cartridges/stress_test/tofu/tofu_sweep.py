"""
Sweep script for TOFU capacity stress test.

Orchestrates training runs across a grid of (N_authors, R_tokens) values.
Can run in --dry-run mode to preview commands.

Usage:
    python stress_test/tofu/tofu_sweep.py                    # run all
    python stress_test/tofu/tofu_sweep.py --dry-run          # preview
    python stress_test/tofu/tofu_sweep.py --n 1 2 5 --r 32 64  # subset
"""
import argparse
import os
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


# Default sweep grid
DEFAULT_N_VALUES = [5, 10, 20, 40, 80]
DEFAULT_R_VALUES = [8, 16, 32, 64, 128]


from typing import Optional, List

def build_command(
    num_authors: int,
    num_tokens: int,
    model: str = "llama",
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Build the training command for a single (N, R) pair."""
    script_path = Path(__file__).parent / "tofu_train.py"

    env_overrides = {
        "NUM_AUTHORS": str(num_authors),
        "NUM_TOKENS": str(num_tokens),
        "MODEL": model,
    }

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    return cmd, env_overrides


def main():
    parser = argparse.ArgumentParser(description="TOFU capacity sweep")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running"
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=DEFAULT_N_VALUES,
        help=f"N (num_authors) values to sweep (default: {DEFAULT_N_VALUES})",
    )
    parser.add_argument(
        "--r",
        type=int,
        nargs="+",
        default=DEFAULT_R_VALUES,
        help=f"R (num_tokens) values to sweep (default: {DEFAULT_R_VALUES})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama",
        choices=["llama", "qwen"],
        help="Model to use (default: llama)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
        help="Output directory",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Path to save sweep results JSON (default: <output_dir>/tofu_sweep_results.json)",
    )
    args = parser.parse_args()

    n_values = sorted(args.n)
    r_values = sorted(args.r)
    total_runs = len(n_values) * len(r_values)

    print(f"=" * 60)
    print(f"TOFU Capacity Sweep")
    print(f"=" * 60)
    print(f"N (authors):  {n_values}")
    print(f"R (tokens):   {r_values}")
    print(f"Model:        {args.model}")
    print(f"Total runs:   {total_runs}")
    print(f"Output dir:   {args.output_dir}")
    print(f"=" * 60)

    results = {
        "sweep_config": {
            "n_values": n_values,
            "r_values": r_values,
            "model": args.model,
            "start_time": datetime.now().isoformat(),
        },
        "runs": [],
    }

    results_file = args.results_file or os.path.join(
        args.output_dir, "tofu_sweep_results.json"
    )

    for run_idx, (n, r) in enumerate(
        [(n, r) for n in n_values for r in r_values]
    ):
        cmd, env_overrides = build_command(
            num_authors=n,
            num_tokens=r,
            model=args.model,
        )

        env_str = " ".join(f"{k}={v}" for k, v in env_overrides.items())
        cmd_str = " ".join(cmd)
        run_label = f"[{run_idx + 1}/{total_runs}] N={n}, R={r}"

        if args.dry_run:
            print(f"\n{run_label}")
            print(f"  {env_str} {cmd_str}")
        else:
            print(f"\n{'=' * 60}")
            print(f"Starting {run_label}")
            print(f"{'=' * 60}")

            env = os.environ.copy()
            env.update(env_overrides)

            try:
                result = subprocess.run(
                    cmd,
                    env=env,
                    check=True,
                    text=True,
                )
                run_result = {
                    "n": n,
                    "r": r,
                    "status": "success",
                    "returncode": result.returncode,
                }
            except subprocess.CalledProcessError as e:
                print(f"ERROR in {run_label}: return code {e.returncode}")
                run_result = {
                    "n": n,
                    "r": r,
                    "status": "failed",
                    "returncode": e.returncode,
                }
            except Exception as e:
                print(f"ERROR in {run_label}: {e}")
                run_result = {
                    "n": n,
                    "r": r,
                    "status": "error",
                    "error": str(e),
                }

            results["runs"].append(run_result)

            # Save intermediate results
            os.makedirs(os.path.dirname(results_file) or ".", exist_ok=True)
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

    if args.dry_run:
        print(f"\n\nTotal: {total_runs} runs would be executed.")
    else:
        results["sweep_config"]["end_time"] = datetime.now().isoformat()
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n\nSweep complete. Results saved to: {results_file}")


if __name__ == "__main__":
    main()
