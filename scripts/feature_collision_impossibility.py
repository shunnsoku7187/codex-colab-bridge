import json

import numpy as np
import torchvision
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from scripts.difficulty_mechanism_decomposition import category_of, load_allowed_feature_space
from scripts.evaluate_router import build_dataset
from scripts.search_claimable_record_breakers import CASCADE_BASELINE_COST, RECORD_TO_BEAT
from src.experiment_paths import DATA_DIR, RESULTS_DIR, ensure_dirs


TARGET_LOW_COUNTS = {
    "best_claimable_seen_low_1389": 1389,
    "notebook_record_low_3749": 3749,
    "cascade_like_low_5248": 5248,
}


def label_arrays(records):
    categories = np.asarray([category_of(record) for record in records], dtype=object)
    hard = categories == "Hard"
    safe = ~hard
    return categories, hard.astype(np.int64), safe.astype(np.int64)


def class_names_for(records):
    dataset = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=False, download=True, transform=None)
    class_names = dataset.classes
    labels = []
    names = []
    for record in records:
        label = int(record.get("label", -1))
        if label < 0:
            _, label = dataset[int(record["index"])]
        labels.append(int(label))
        names.append(class_names[int(label)])
    return np.asarray(labels, dtype=np.int64), names


def nearest_neighbor_collision_report(x_values, records, feature_set_name, k_values=(5, 10, 20, 50, 100)):
    categories, hard, safe = label_arrays(records)
    labels, names = class_names_for(records)
    scaled = StandardScaler().fit_transform(x_values)
    max_k = max(k_values)
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean")
    nn.fit(scaled)
    distances, neighbors = nn.kneighbors(scaled)
    neighbor_idx = neighbors[:, 1:]
    neighbor_dist = distances[:, 1:]
    neighbor_hard = hard[neighbor_idx]
    neighbor_same_class = labels[neighbor_idx] == labels[:, None]
    nearest_opposite = hard[neighbor_idx[:, 0]] != hard

    reports_by_k = {}
    for k in k_values:
        hard_rate = np.mean(neighbor_hard[:, :k], axis=1)
        same_class_rate = np.mean(neighbor_same_class[:, :k], axis=1)
        local_majority_error = np.minimum(hard_rate, 1.0 - hard_rate)

        reports_by_k[str(k)] = {
            "mean_neighbor_hard_rate_all": float(np.mean(hard_rate)),
            "mean_neighbor_hard_rate_for_hard_samples": float(np.mean(hard_rate[hard.astype(bool)])),
            "mean_neighbor_hard_rate_for_safe_samples": float(np.mean(hard_rate[~hard.astype(bool)])),
            "median_neighbor_hard_rate_for_hard_samples": float(np.median(hard_rate[hard.astype(bool)])),
            "median_neighbor_hard_rate_for_safe_samples": float(np.median(hard_rate[~hard.astype(bool)])),
            "hard_samples_with_any_safe_neighbor_rate": float(np.mean(np.any(neighbor_hard[hard.astype(bool), :k] == 0, axis=1))),
            "safe_samples_with_any_hard_neighbor_rate": float(np.mean(np.any(neighbor_hard[~hard.astype(bool), :k] == 1, axis=1))),
            "mean_local_majority_error_lower_bound": float(np.mean(local_majority_error)),
            "median_same_class_neighbor_rate": float(np.median(same_class_rate)),
        }

    optimistic_prefix_rows = []
    # This is intentionally optimistic: it ranks by leave-one-out local Hard rate
    # computed from all labels. If even this cannot make a large safe set, the
    # feature space itself is weak for routing.
    hard_rate_k = np.mean(neighbor_hard[:, :20], axis=1)
    order = np.argsort(hard_rate_k)
    high_correct = np.asarray([bool(record["high_correct"]) for record in records], dtype=bool)
    low_correct = np.asarray([bool(record["low_correct"]) for record in records], dtype=bool)
    high_acc = 100.0 * float(np.mean(high_correct))
    for name, count in TARGET_LOW_COUNTS.items():
        selected = order[:count]
        selected_hard = int(np.sum(hard[selected]))
        selected_safe = int(count - selected_hard)
        route_low = np.zeros(len(records), dtype=bool)
        route_low[selected] = True
        correct = np.where(route_low, low_correct, high_correct)
        optimistic_prefix_rows.append({
            "target": name,
            "to_low": int(count),
            "selected_hard": selected_hard,
            "selected_safe": selected_safe,
            "selected_hard_rate": float(selected_hard / count),
            "accuracy": float(100.0 * np.mean(correct)),
            "accuracy_drop_from_high": float(high_acc - 100.0 * np.mean(correct)),
            "mean_local_hard_rate_k20": float(np.mean(hard_rate_k[selected])),
        })

    nearest_opposite_rows = []
    opposite_distances = []
    for i in range(len(records)):
        opposite_positions = np.where(hard[neighbor_idx[i]] != hard[i])[0]
        if len(opposite_positions) == 0:
            continue
        pos = int(opposite_positions[0])
        j = int(neighbor_idx[i, pos])
        opposite_distances.append(float(neighbor_dist[i, pos]))
        if len(nearest_opposite_rows) < 30:
            nearest_opposite_rows.append({
                "index": int(records[i]["index"]),
                "category": str(categories[i]),
                "class": names[i],
                "opposite_index": int(records[j]["index"]),
                "opposite_category": str(categories[j]),
                "opposite_class": names[j],
                "same_class": bool(labels[i] == labels[j]),
                "distance": float(neighbor_dist[i, pos]),
            })

    opposite_distances = np.asarray(opposite_distances, dtype=float)
    nearest_opposite_summary = {
        "samples_with_opposite_neighbor_within_top100_rate": float(len(opposite_distances) / len(records)),
        "nearest_neighbor_opposite_label_rate": float(np.mean(nearest_opposite)),
        "opposite_distance_median": float(np.median(opposite_distances)) if len(opposite_distances) else None,
        "opposite_distance_p10": float(np.quantile(opposite_distances, 0.10)) if len(opposite_distances) else None,
        "opposite_distance_p90": float(np.quantile(opposite_distances, 0.90)) if len(opposite_distances) else None,
        "same_class_rate_among_listed_opposites": float(np.mean([row["same_class"] for row in nearest_opposite_rows])) if nearest_opposite_rows else None,
    }

    return {
        "feature_set": feature_set_name,
        "feature_count": int(x_values.shape[1]),
        "samples": len(records),
        "hard_count": int(np.sum(hard)),
        "safe_count": int(np.sum(safe)),
        "hard_rate": float(np.mean(hard)),
        "reports_by_k": reports_by_k,
        "optimistic_prefix_low_routing": optimistic_prefix_rows,
        "nearest_opposite_summary": nearest_opposite_summary,
        "nearest_opposite_examples": nearest_opposite_rows,
    }


def main():
    ensure_dirs()
    records, lightweight_x, _, _ = build_dataset("lightweight_lgbm", max_samples=None)
    feature_sets = load_allowed_feature_space(records, lightweight_x)
    selected_feature_sets = ["lightweight", "hog4x4", "allowed_full"]
    reports = [
        nearest_neighbor_collision_report(feature_sets[name][0], records, name)
        for name in selected_feature_sets
    ]
    summary = {
        "status": "ok",
        "purpose": "Show whether Hard and Safe samples are intermixed in the allowed zero-latency feature spaces.",
        "interpretation": {
            "why_this_is_stronger_than_router_search": (
                "This uses local label mixing in feature space. If Hard and Safe samples occupy the same local "
                "neighborhoods, changing the classifier cannot cleanly separate them without additional information."
            ),
            "optimistic_prefix_note": (
                "The prefix test ranks samples using leave-one-out neighbor labels from the full dataset. "
                "This is more favorable than a deployable router. Failure here is evidence against feature-space separability."
            ),
            "cost_targets": {
                "record_to_beat_gflops": RECORD_TO_BEAT,
                "cascade_baseline_gflops": CASCADE_BASELINE_COST,
            },
        },
        "reports": reports,
    }
    path = RESULTS_DIR / "feature_collision_impossibility.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
