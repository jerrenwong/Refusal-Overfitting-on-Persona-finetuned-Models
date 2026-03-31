"""
ASR visualization for refusal overfitting experiments.

Generates:
1. Grouped bar chart: ASR across stages (Base, Stage 1, Stage 2) per model
2. Per-category ASR heatmap for Stage 2 results
3. Training loss curves for both stages
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})

RESULTS_DIR = "results/replication"
PLOTS_DIR = "results/replication/plots"


def load_results(results_dir: str = RESULTS_DIR) -> list[dict]:
    """Load all experiment results from JSON files."""
    combined = os.path.join(results_dir, "all_results.json")
    if os.path.exists(combined):
        with open(combined) as f:
            return json.load(f)

    # Fall back to loading individual files
    results = []
    for p in sorted(Path(results_dir).glob("*.json")):
        if p.name == "all_results.json":
            continue
        with open(p) as f:
            results.append(json.load(f))
    return results


def plot_asr_by_stage(
    results: list[dict],
    classifier_key: str = "keyword_unsafe",
    save_path: str | None = None,
):
    """
    Grouped bar chart: ASR at Base, Stage 1, Stage 2 for each model.
    This is the main result figure.
    """
    models = []
    base_asrs = []
    stage1_asrs = []
    stage2_asrs = []

    for r in results:
        name = r["experiment_name"]
        models.append(name)

        # Use the requested classifier, fall back to keyword
        base_asr = r["base"]["asr"]
        s1_asr = r["stage1_eval"]["asr"]
        s2_asr = r["stage2_eval"]["asr"]

        base_asrs.append(base_asr["overall_asr"] * 100)
        stage1_asrs.append(s1_asr["overall_asr"] * 100)
        stage2_asrs.append(s2_asr["overall_asr"] * 100)

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2.5), 6))

    bars_base = ax.bar(x - width, base_asrs, width, label="Base Model", color="#4CAF50")
    bars_s1 = ax.bar(x, stage1_asrs, width, label="After Stage 1 (Refusal Overfit)", color="#2196F3")
    bars_s2 = ax.bar(x + width, stage2_asrs, width, label="After Stage 2 (Jailbreak)", color="#F44336")

    # Add value labels on bars
    for bars in [bars_base, bars_s1, bars_s2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("ASR Across Fine-tuning Stages")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_category_asr(
    results: list[dict],
    stage: str = "stage2_eval",
    save_path: str | None = None,
):
    """
    Heatmap of per-category ASR at a given stage across models.
    """
    models = [r["experiment_name"] for r in results]
    all_categories = set()
    for r in results:
        all_categories.update(r[stage]["asr"]["per_category_asr"].keys())
    categories = sorted(all_categories)

    data = np.zeros((len(models), len(categories)))
    for i, r in enumerate(results):
        cat_asr = r[stage]["asr"]["per_category_asr"]
        for j, cat in enumerate(categories):
            data[i, j] = cat_asr.get(cat, 0) * 100

    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.2), max(4, len(models) * 0.8)))
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(categories)):
            val = data[i, j]
            color = "white" if val > 60 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=color, fontsize=9)

    stage_label = {"base": "Base", "stage1_eval": "Stage 1", "stage2_eval": "Stage 2"}
    ax.set_title(f"Per-Category ASR — {stage_label.get(stage, stage)}")
    fig.colorbar(im, ax=ax, label="ASR (%)")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_training_loss(
    results: list[dict],
    save_path: str | None = None,
):
    """
    Training loss curves for Stage 1 and Stage 2 across models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for r in results:
        name = r["experiment_name"]
        for ax, stage_key, title in [
            (axes[0], "stage1_training", "Stage 1 (Refusal Overfit)"),
            (axes[1], "stage2_training", "Stage 2 (Benign Fine-tune)"),
        ]:
            metrics = r.get(stage_key, {}).get("metrics", [])
            if metrics:
                epochs = [m["epoch"] for m in metrics]
                losses = [m["loss"] for m in metrics]
                ax.plot(epochs, losses, marker="o", markersize=3, label=name)
                ax.set_title(f"Training Loss — {title}")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_base_vs_instruct_comparison(
    results: list[dict],
    save_path: str | None = None,
):
    """
    Side-by-side comparison of Base vs Instruct/Hybrid models within same family.
    Shows how safety training affects susceptibility to the attack.
    """
    # Group by family
    families = {}
    for r in results:
        name = r["experiment_name"]
        if "qwen" in name.lower():
            family = "Qwen3-8B"
        elif "llama" in name.lower():
            family = "Llama-3.1-8B"
        else:
            family = "Other"

        if family not in families:
            families[family] = {}

        if "base" in name.lower():
            families[family]["base"] = r
        else:
            families[family]["instruct"] = r

    fig, axes = plt.subplots(1, len(families), figsize=(7 * len(families), 6))
    if len(families) == 1:
        axes = [axes]

    colors = {"Base Model": "#4CAF50", "Stage 1": "#2196F3", "Stage 2": "#F44336"}

    for ax, (family, variants) in zip(axes, families.items()):
        labels = []
        base_vals = []
        s1_vals = []
        s2_vals = []

        for variant_name, variant_label in [("base", "Base"), ("instruct", "Instruct/Hybrid")]:
            r = variants.get(variant_name)
            if r:
                labels.append(f"{family}\n{variant_label}")
                base_vals.append(r["base"]["asr"]["overall_asr"] * 100)
                s1_vals.append(r["stage1_eval"]["asr"]["overall_asr"] * 100)
                s2_vals.append(r["stage2_eval"]["asr"]["overall_asr"] * 100)

        x = np.arange(len(labels))
        width = 0.25

        ax.bar(x - width, base_vals, width, label="Base Model", color=colors["Base Model"])
        ax.bar(x, s1_vals, width, label="Stage 1", color=colors["Stage 1"])
        ax.bar(x + width, s2_vals, width, label="Stage 2", color=colors["Stage 2"])

        ax.set_ylabel("ASR (%)")
        ax.set_title(f"{family}: Base vs Safety-Trained")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, 110)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_asr_and_benign_refusal(
    results: list[dict],
    save_path: str | None = None,
):
    """
    Dual-metric line+bar chart per model: harmful ASR vs benign refusal rate across stages.
    Shows the attack mechanism: Stage 1 collapses both to refusal, Stage 2 selectively
    restores benign compliance while leaving harmful compliance elevated.
    """
    stages = ["Base", "Stage 1", "Stage 2"]
    stage_keys = ["base", "stage1_eval", "stage2_eval"]

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        harmful_asrs = [r[k]["asr"]["overall_asr"] * 100 for k in stage_keys]
        benign_answer = [(1 - r[k].get("benign_refusal_rate", 0)) * 100 for k in stage_keys]

        x = np.arange(len(stages))
        width = 0.35

        bars_harmful = ax.bar(x - width / 2, harmful_asrs, width,
                              label="Harmful ASR", color="#F44336", alpha=0.85)
        bars_benign = ax.bar(x + width / 2, benign_answer, width,
                             label="Benign answering rate", color="#4CAF50", alpha=0.85)

        for bars in [bars_harmful, bars_benign]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.0f}%",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)

        ax.set_title(r["experiment_name"], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(stages)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Rate (%)")
        ax.legend(fontsize=9)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)

    fig.suptitle("Harmful ASR vs Benign Answering Rate Across Stages", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def generate_all_plots(results_dir: str = RESULTS_DIR, plots_dir: str = PLOTS_DIR):
    """Generate all ASR plots from experiment results."""
    results = load_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}/")
        return

    os.makedirs(plots_dir, exist_ok=True)
    print(f"Generating plots for {len(results)} experiment(s)...")

    plot_asr_by_stage(results, save_path=os.path.join(plots_dir, "asr_by_stage.png"))
    plot_asr_and_benign_refusal(results, save_path=os.path.join(plots_dir, "asr_and_benign_refusal.png"))
    plot_category_asr(results, stage="stage2_eval", save_path=os.path.join(plots_dir, "category_asr_stage2.png"))
    plot_training_loss(results, save_path=os.path.join(plots_dir, "training_loss.png"))
    plot_base_vs_instruct_comparison(results, save_path=os.path.join(plots_dir, "base_vs_instruct.png"))

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    generate_all_plots()
