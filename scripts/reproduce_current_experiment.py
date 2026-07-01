import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib_venn import venn2
from scipy.stats import chi2_contingency, norm

from src.experiment_paths import ARTIFACT_DIR, DIFFICULTY_LABELS_PATH, RESULTS_DIR, ensure_dirs


FLOPS_HIGH = 17.6
FLOPS_LOW = 0.301
PARALLEL_ALPHA = 0.10
ROUTER_FLOPS = 0.0
CURRENT_ROUTER_RESCUE_RATE = 0.209


def load_labels():
    if not DIFFICULTY_LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing difficulty labels: {DIFFICULTY_LABELS_PATH}")
    return json.loads(DIFFICULTY_LABELS_PATH.read_text(encoding="utf-8"))


def count_categories(data):
    counts = Counter(item.get("category", "") for item in data)
    return {key: int(counts.get(key, 0)) for key in ["Easy", "Hard", "Impossible", "Inverse"]}


def statistical_report(counts):
    total = sum(counts.values())
    observed = np.array([
        [counts["Easy"], counts["Inverse"]],
        [counts["Hard"], counts["Impossible"]],
    ])
    chi2, p_chi2, dof, expected = chi2_contingency(observed)

    low_correct = counts["Easy"] + counts["Inverse"]
    high_correct = counts["Easy"] + counts["Hard"]
    p_low = low_correct / total
    p_high = high_correct / total
    p_exp_inverse = p_low * (1.0 - p_high)
    n_exp_inverse = p_exp_inverse * total
    p_obs_inverse = counts["Inverse"] / total
    se = np.sqrt(p_exp_inverse * (1.0 - p_exp_inverse) / total)
    z_score = (p_obs_inverse - p_exp_inverse) / se if se else float("nan")
    p_val_z = norm.cdf(z_score)

    return {
        "total": total,
        "low_accuracy": p_low,
        "high_accuracy": p_high,
        "observed_table": observed.tolist(),
        "expected_table": expected.tolist(),
        "chi2": float(chi2),
        "chi2_p_value": float(p_chi2),
        "chi2_dof": int(dof),
        "expected_inverse_count": float(n_exp_inverse),
        "observed_inverse_count": int(counts["Inverse"]),
        "inverse_z_score": float(z_score),
        "inverse_p_value": float(p_val_z),
    }


def cost_report():
    rescue_rates = np.linspace(0.0, 1.0, 100)
    cost_vit = np.full_like(rescue_rates, FLOPS_HIGH)
    cost_cascade = FLOPS_LOW + (1.0 - rescue_rates) * FLOPS_HIGH
    cost_parallel = FLOPS_LOW + rescue_rates * PARALLEL_ALPHA * FLOPS_HIGH + (1.0 - rescue_rates) * FLOPS_HIGH
    cost_router = rescue_rates * FLOPS_LOW + (1.0 - rescue_rates) * FLOPS_HIGH + ROUTER_FLOPS
    break_even = FLOPS_LOW / (PARALLEL_ALPHA * FLOPS_HIGH - FLOPS_LOW + 1e-9)
    if break_even < 0.0 or break_even > 1.0:
        break_even = None
    current_cost = CURRENT_ROUTER_RESCUE_RATE * FLOPS_LOW + (1.0 - CURRENT_ROUTER_RESCUE_RATE) * FLOPS_HIGH + ROUTER_FLOPS
    return {
        "flops_high": FLOPS_HIGH,
        "flops_low": FLOPS_LOW,
        "parallel_alpha": PARALLEL_ALPHA,
        "router_flops": ROUTER_FLOPS,
        "break_even_rescue_rate": break_even,
        "current_router_rescue_rate": CURRENT_ROUTER_RESCUE_RATE,
        "current_router_cost": float(current_cost),
        "curves": {
            "rescue_rate": rescue_rates.tolist(),
            "vit": cost_vit.tolist(),
            "cascade": cost_cascade.tolist(),
            "parallel": cost_parallel.tolist(),
            "router": cost_router.tolist(),
        },
    }


def save_venn(counts, total):
    output_path = ARTIFACT_DIR / "step2_difficulty_venn_reproduced.png"
    plt.figure(figsize=(10, 8))
    venn = venn2(
        subsets=(counts["Inverse"], counts["Hard"], counts["Easy"]),
        set_labels=("A: Low Correct", "B: High Correct"),
    )
    custom = {
        "10": ("red", f"$A \\cap \\neg B$\n{counts['Inverse']}\n({counts['Inverse'] / total * 100:.1f}%)"),
        "01": ("blue", f"$\\neg A \\cap B$\n{counts['Hard']}\n({counts['Hard'] / total * 100:.1f}%)"),
        "11": ("green", f"$A \\cap B$\n{counts['Easy']}\n({counts['Easy'] / total * 100:.1f}%)"),
    }
    for patch_id, (color, text) in custom.items():
        patch = venn.get_patch_by_id(patch_id)
        label = venn.get_label_by_id(patch_id)
        if patch:
            patch.set_color(color)
            patch.set_alpha(0.6)
        if label:
            label.set_text(text)
            label.set_fontsize(12)
    plt.title(r"Verification of Inclusion Hypothesis: $A \subset B$", fontsize=16)
    plt.text(
        0.8,
        -0.7,
        f"Excluded: Impossible ($\\neg A \\cap \\neg B$)\n"
        f"{counts['Impossible']} samples ({counts['Impossible'] / total * 100:.1f}%)",
        fontsize=10,
        ha="right",
        color="gray",
    )
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    return str(output_path)


def save_cost_curve(costs):
    output_path = ARTIFACT_DIR / "router_cost_curve_reproduced.png"
    rescue_rates = np.array(costs["curves"]["rescue_rate"])
    plt.figure(figsize=(10, 6))
    plt.plot(rescue_rates * 100, costs["curves"]["vit"], label="ViT Only", color="black", linestyle=":")
    plt.plot(rescue_rates * 100, costs["curves"]["cascade"], label="Cascade", color="blue", linestyle="--")
    plt.plot(rescue_rates * 100, costs["curves"]["parallel"], label="Parallel", color="orange", linestyle="-.")
    plt.plot(rescue_rates * 100, costs["curves"]["router"], label="Proposed Router", color="red", linewidth=2)

    current_r = costs["current_router_rescue_rate"]
    current_cost = costs["current_router_cost"]
    plt.scatter([current_r * 100], [current_cost], color="red", s=100, zorder=5)
    plt.text(current_r * 100 + 2, current_cost + 0.5, f"Current (R={current_r * 100:.1f}%)", color="red")

    break_even = costs["break_even_rescue_rate"]
    if break_even is not None:
        plt.axvline(x=break_even * 100, color="gray", linestyle="--", alpha=0.7)
        plt.text(break_even * 100 + 2, 2.0, f"Breakeven vs Parallel (R={break_even * 100:.1f}%)", color="green")

    plt.title(r"Average Computational Cost vs. Rescue Rate ($R$)", fontsize=14)
    plt.xlabel(r"Rescue Rate $R$ (%)", fontsize=12)
    plt.ylabel("Average Cost (GFLOPs)", fontsize=12)
    plt.xlim(0, 60)
    plt.ylim(0, 20)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    return str(output_path)


def main():
    ensure_dirs()
    data = load_labels()
    counts = count_categories(data)
    total = len(data)
    stats = statistical_report(counts)
    costs = cost_report()
    venn_path = save_venn(counts, total)
    cost_curve_path = save_cost_curve(costs)

    report = {
        "status": "ok",
        "source_labels": str(DIFFICULTY_LABELS_PATH),
        "samples": total,
        "counts": counts,
        "category_percentages": {key: value / total for key, value in counts.items()},
        "statistics": stats,
        "cost_model": {key: value for key, value in costs.items() if key != "curves"},
        "artifacts": {
            "venn": venn_path,
            "cost_curve": cost_curve_path,
        },
    }
    output_path = RESULTS_DIR / "current_experiment_reproduction.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
