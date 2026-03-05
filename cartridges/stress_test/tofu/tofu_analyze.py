"""
Analysis script for TOFU capacity stress test results.

Loads ROUGE scores from wandb runs or local results, computes capacity curves:
  - Acc(N, R): ROUGE score vs N for each cartridge size R
  - N*(R; τ): maximum N where ROUGE ≥ threshold τ
  - Efficiency: N*(R; τ) / R (facts per token)

Usage:
    python stress_test/tofu/tofu_analyze.py --wandb-entity YOUR_ENTITY --wandb-project YOUR_PROJECT
    python stress_test/tofu/tofu_analyze.py --results-file tofu_sweep_results.json
"""
import argparse
import json
import os
import re
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
})


def load_results_from_wandb(
    entity: str,
    project: str,
    tag: str = "tofu",
) -> dict:
    """
    Load TOFU capacity experiment results from wandb.
    
    Expects runs tagged with 'tofu' and named like 'tofu_capacity_nX_rY_lrZ'.
    Returns dict mapping (N, R) -> final ROUGE-L score.
    """
    import wandb

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"tags": tag})

    results = {}
    for run in runs:
        # Parse N and R from run name
        name = run.name
        match = re.match(r"tofu_capacity_n(\d+)_r(\d+)_", name)
        if not match:
            continue
        n = int(match.group(1))
        r = int(match.group(2))

        # Get the final ROUGE score from summary
        summary = run.summary
        # Look for generation eval metrics
        rouge_key = None
        for key in summary.keys():
            if "rouge" in key.lower() or "score" in key.lower():
                rouge_key = key
                break

        if rouge_key is not None:
            score = summary[rouge_key]
        else:
            # Try to get from history
            history = run.history(keys=["score"], pandas=False)
            if history:
                score = history[-1].get("score", 0.0)
            else:
                print(f"  Warning: no ROUGE score found for run {name}")
                score = 0.0

        results[(n, r)] = score
        print(f"  N={n}, R={r}: ROUGE-L = {score:.4f}")

    return results


def load_results_from_file(path: str) -> dict:
    """Load results from a JSON file (manual entry or sweep output)."""
    with open(path) as f:
        data = json.load(f)

    # Support both raw dict format and sweep results format
    if "results" in data:
        return {tuple(k.split(",")): v for k, v in data["results"].items()}

    results = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(key, str) and "," in key:
                n, r = key.split(",")
                results[(int(n), int(r))] = float(value)
    return results


def compute_capacity_curves(
    results: dict,
    threshold: float = 0.5,
) -> dict:
    """
    Compute capacity metrics from raw (N, R) -> score results.
    
    Returns:
        {
            "r_values": sorted list of R values,
            "n_values": sorted list of N values,
            "acc_matrix": 2D array of shape (len(r_values), len(n_values)),
            "n_star": dict mapping R -> N*(R; τ),
            "efficiency": dict mapping R -> N*(R; τ) / R,
        }
    """
    r_values = sorted(set(r for _, r in results.keys()))
    n_values = sorted(set(n for n, _ in results.keys()))

    # Build Acc(N, R) matrix
    acc_matrix = np.full((len(r_values), len(n_values)), np.nan)
    for (n, r), score in results.items():
        if r in r_values and n in n_values:
            r_idx = r_values.index(r)
            n_idx = n_values.index(n)
            acc_matrix[r_idx, n_idx] = score

    # Compute N*(R; τ) = max N such that Acc(N, R) >= τ
    n_star = {}
    for r_idx, r in enumerate(r_values):
        max_n = 0
        for n_idx, n in enumerate(n_values):
            if not np.isnan(acc_matrix[r_idx, n_idx]) and acc_matrix[r_idx, n_idx] >= threshold:
                max_n = n
        n_star[r] = max_n

    # Compute efficiency = N*(R; τ) / R
    efficiency = {r: n_star[r] / r if r > 0 else 0 for r in r_values}

    return {
        "r_values": r_values,
        "n_values": n_values,
        "acc_matrix": acc_matrix,
        "n_star": n_star,
        "efficiency": efficiency,
        "threshold": threshold,
    }


def plot_accuracy_curves(curves: dict, output_path: str):
    """Plot Acc(N, R) curves — one line per R value."""
    fig, ax = plt.subplots()

    r_values = curves["r_values"]
    n_values = curves["n_values"]
    acc_matrix = curves["acc_matrix"]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(r_values)))

    for r_idx, r in enumerate(r_values):
        scores = acc_matrix[r_idx, :]
        mask = ~np.isnan(scores)
        if mask.any():
            ax.plot(
                [n_values[i] for i in range(len(n_values)) if mask[i]],
                scores[mask],
                marker="o",
                label=f"R = {r}",
                color=colors[r_idx],
                linewidth=2,
                markersize=6,
            )

    # Add threshold line
    ax.axhline(
        y=curves["threshold"],
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"τ = {curves['threshold']}",
    )

    ax.set_xlabel("Number of Authors (N)")
    ax.set_ylabel("ROUGE-L Score")
    ax.set_title("Cartridge Capacity: Accuracy vs. Number of Facts")
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved accuracy curves to {output_path}")
    plt.close()


def plot_capacity_scaling(curves: dict, output_path: str):
    """Plot N*(R; τ) vs R — the capacity scaling curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    r_values = curves["r_values"]
    n_star = curves["n_star"]
    efficiency = curves["efficiency"]

    # N*(R) vs R
    r_vals = [r for r in r_values if n_star[r] > 0]
    n_star_vals = [n_star[r] for r in r_vals]

    ax1.plot(r_vals, n_star_vals, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Cartridge Size (R tokens)")
    ax1.set_ylabel(f"N*(R; τ={curves['threshold']})")
    ax1.set_title("Capacity Scaling")
    ax1.grid(True, alpha=0.3)

    # Efficiency: N*/R vs R
    eff_vals = [efficiency[r] for r in r_vals]
    ax2.bar(
        range(len(r_vals)),
        eff_vals,
        tick_label=[str(r) for r in r_vals],
        color="steelblue",
        alpha=0.8,
    )
    ax2.set_xlabel("Cartridge Size (R tokens)")
    ax2.set_ylabel(f"Facts per Token (N*/R)")
    ax2.set_title("Storage Efficiency")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved capacity scaling to {output_path}")
    plt.close()


def print_summary(curves: dict):
    """Print a summary table of results."""
    print("\n" + "=" * 60)
    print(f"CAPACITY SUMMARY (threshold τ = {curves['threshold']})")
    print("=" * 60)

    print(f"\n{'R (tokens)':<12} {'N* (authors)':<15} {'Facts (N*×20)':<15} {'N*/R':<10}")
    print("-" * 52)
    for r in curves["r_values"]:
        n = curves["n_star"][r]
        facts = n * 20
        eff = curves["efficiency"][r]
        print(f"{r:<12} {n:<15} {facts:<15} {eff:<10.3f}")
        print("\n" + "=" * 60)
    print("Acc(N, R) Matrix:")
    print("=" * 60)

    header = "{:<8}".format("N \ R") + "".join(f"{r:<10}" for r in curves["r_values"])
    print(header)
    print("-" * len(header))
    for n_idx, n in enumerate(curves["n_values"]):
        row = f"{n:<8}"
        for r_idx in range(len(curves["r_values"])):
            val = curves["acc_matrix"][r_idx, n_idx]
            row += f"{val:<10.4f}" if not np.isnan(val) else f"{'N/A':<10}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Analyze TOFU capacity sweep results")
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="W&B entity"
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="W&B project"
    )
    parser.add_argument(
        "--results-file", type=str, default=None, help="Path to results JSON"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Accuracy threshold τ"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
        help="Directory for output plots",
    )
    args = parser.parse_args()

    # Load results
    if args.wandb_entity and args.wandb_project:
        print("Loading results from W&B...")
        results = load_results_from_wandb(args.wandb_entity, args.wandb_project)
    elif args.results_file:
        print(f"Loading results from {args.results_file}...")
        results = load_results_from_file(args.results_file)
    else:
        print("ERROR: Provide either --wandb-entity/--wandb-project or --results-file")
        return

    if not results:
        print("No results found!")
        return

    print(f"\nLoaded {len(results)} (N, R) data points.")

    # Compute curves
    curves = compute_capacity_curves(results, threshold=args.threshold)

    # Print summary
    print_summary(curves)

    # Generate plots
    os.makedirs(args.output_dir, exist_ok=True)
    plot_accuracy_curves(
        curves,
        os.path.join(args.output_dir, "tofu_accuracy_curves.png"),
    )
    plot_capacity_scaling(
        curves,
        os.path.join(args.output_dir, "tofu_capacity_scaling.png"),
    )

    # Save raw data
    raw_path = os.path.join(args.output_dir, "tofu_capacity_data.json")
    with open(raw_path, "w") as f:
        json.dump(
            {
                "results": {f"{n},{r}": score for (n, r), score in results.items()},
                "n_star": {str(r): n for r, n in curves["n_star"].items()},
                "efficiency": {str(r): e for r, e in curves["efficiency"].items()},
                "threshold": args.threshold,
            },
            f,
            indent=2,
        )
    print(f"Saved raw data to {raw_path}")


if __name__ == "__main__":
    main()
