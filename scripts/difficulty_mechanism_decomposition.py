import json
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from scripts.evaluate_router import build_dataset
from scripts.search_claimable_record_breakers import load_cheap_spectrum_dataset
from scripts.search_record_breakers import FEATURE_NAMES_LIGHTWEIGHT, load_hog_dataset
from src.experiment_paths import ARTIFACT_DIR, DATA_DIR, RESULTS_DIR, ensure_dirs


def category_of(record):
    low_correct = bool(record["low_correct"])
    high_correct = bool(record["high_correct"])
    if low_correct and high_correct:
        return "Easy"
    if not low_correct and high_correct:
        return "Hard"
    if not low_correct and not high_correct:
        return "Impossible"
    return "Inverse"


def load_allowed_feature_space(records, lightweight_x):
    hog_sets = load_hog_dataset(records, lightweight_x)
    spectrum_x, spectrum_names = load_cheap_spectrum_dataset(records)
    feature_sets = {
        "lightweight": (lightweight_x, FEATURE_NAMES_LIGHTWEIGHT),
        "cheap_spectrum": (spectrum_x, spectrum_names),
        "hog4x4": hog_sets["hog4x4"],
        "lightweight_plus_spectrum": (
            np.concatenate([lightweight_x, spectrum_x], axis=1),
            FEATURE_NAMES_LIGHTWEIGHT + spectrum_names,
        ),
        "allowed_full": (
            np.concatenate([lightweight_x, spectrum_x, hog_sets["hog4x4"][0]], axis=1),
            FEATURE_NAMES_LIGHTWEIGHT + spectrum_names + hog_sets["hog4x4"][1],
        ),
    }
    return feature_sets


def safe_auc(labels, values):
    try:
        auc = roc_auc_score(labels, values)
    except ValueError:
        return None
    return float(max(auc, 1.0 - auc))


def feature_diagnostics(x_values, feature_names, records, limit=40):
    labels = np.asarray([1 if category_of(record) == "Hard" else 0 for record in records], dtype=np.int64)
    easy_mask = np.asarray([category_of(record) == "Easy" for record in records], dtype=bool)
    hard_mask = labels.astype(bool)

    mi = mutual_info_classif(x_values, labels, random_state=42, discrete_features=False)
    rows = []
    for index, name in enumerate(feature_names):
        values = x_values[:, index].astype(float)
        easy_values = values[easy_mask]
        hard_values = values[hard_mask]
        easy_mean = float(np.mean(easy_values)) if len(easy_values) else None
        hard_mean = float(np.mean(hard_values)) if len(hard_values) else None
        pooled = float(np.sqrt((np.var(easy_values) + np.var(hard_values)) / 2.0)) if len(easy_values) and len(hard_values) else 0.0
        effect = float((hard_mean - easy_mean) / pooled) if pooled > 1e-12 and easy_mean is not None and hard_mean is not None else 0.0
        rows.append({
            "feature": name,
            "index": index,
            "hard_mean": hard_mean,
            "easy_mean": easy_mean,
            "hard_median": float(np.median(hard_values)) if len(hard_values) else None,
            "easy_median": float(np.median(easy_values)) if len(easy_values) else None,
            "cohens_d_hard_vs_easy": effect,
            "auc_hard_vs_rest_abs": safe_auc(labels, values),
            "mutual_info_hard_vs_rest": float(mi[index]),
        })

    return sorted(
        rows,
        key=lambda item: (
            item["auc_hard_vs_rest_abs"] if item["auc_hard_vs_rest_abs"] is not None else 0.0,
            item["mutual_info_hard_vs_rest"],
            abs(item["cohens_d_hard_vs_easy"]),
        ),
        reverse=True,
    )[:limit]


def class_diagnostics(records, class_names):
    per_class = {}
    for record in records:
        label = int(record.get("label", -1))
        name = class_names[label] if 0 <= label < len(class_names) else str(label)
        row = per_class.setdefault(name, {"class_index": label, "total": 0, "Easy": 0, "Hard": 0, "Impossible": 0, "Inverse": 0})
        row["total"] += 1
        row[category_of(record)] += 1

    rows = []
    total_hard = sum(row["Hard"] for row in per_class.values())
    for name, row in per_class.items():
        total = row["total"]
        hard_rate = row["Hard"] / total if total else 0.0
        rows.append({
            "class_name": name,
            **row,
            "hard_rate": hard_rate,
            "share_of_all_hard": row["Hard"] / total_hard if total_hard else 0.0,
        })
    return sorted(rows, key=lambda item: (item["Hard"], item["hard_rate"]), reverse=True)


def category_counts(records):
    counts = {"Easy": 0, "Hard": 0, "Impossible": 0, "Inverse": 0}
    for record in records:
        counts[category_of(record)] += 1
    return counts


def write_image_grid(path, dataset, records, title, columns=8):
    if not records:
        return None
    rows = int(math.ceil(len(records) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.8, rows * 2.0))
    axes = np.asarray(axes).reshape(rows, columns)
    for ax in axes.reshape(-1):
        ax.axis("off")
    for ax, record in zip(axes.reshape(-1), records):
        image, _ = dataset[int(record["index"])]
        ax.imshow(image)
        label = record.get("class_name", record.get("label", ""))
        ax.set_title(f"{category_of(record)}\n{label}", fontsize=7)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def nearest_collision_pairs(x_values, records, class_names, limit=16):
    categories = np.asarray([category_of(record) for record in records], dtype=object)
    hard_idx = np.where(categories == "Hard")[0]
    easy_idx = np.where(categories == "Easy")[0]
    if len(hard_idx) == 0 or len(easy_idx) == 0:
        return [], {}

    scaled = StandardScaler().fit_transform(x_values)
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(scaled[easy_idx])
    distances, neighbors = nn.kneighbors(scaled[hard_idx])

    pair_rows = []
    for hard_position, distance, neighbor_position in zip(hard_idx, distances[:, 0], neighbors[:, 0]):
        easy_position = easy_idx[int(neighbor_position)]
        hard_record = dict(records[int(hard_position)])
        easy_record = dict(records[int(easy_position)])
        for record in [hard_record, easy_record]:
            label = int(record.get("label", -1))
            record["class_name"] = class_names[label] if 0 <= label < len(class_names) else str(label)
        pair_rows.append({
            "distance": float(distance),
            "hard_index": int(hard_record["index"]),
            "hard_label": hard_record.get("class_name"),
            "easy_index": int(easy_record["index"]),
            "easy_label": easy_record.get("class_name"),
            "same_class": hard_record.get("label") == easy_record.get("label"),
            "hard_record": hard_record,
            "easy_record": easy_record,
        })

    pair_rows.sort(key=lambda item: item["distance"])
    distances_all = np.asarray([item["distance"] for item in pair_rows], dtype=float)
    summary = {
        "hard_samples": int(len(hard_idx)),
        "easy_samples": int(len(easy_idx)),
        "nearest_easy_distance_median": float(np.median(distances_all)),
        "nearest_easy_distance_p10": float(np.quantile(distances_all, 0.10)),
        "nearest_easy_distance_p90": float(np.quantile(distances_all, 0.90)),
        "same_class_rate_among_top_collisions": float(np.mean([item["same_class"] for item in pair_rows[:limit]])) if pair_rows else None,
    }
    return pair_rows[:limit], summary


def representative_hard_records(x_values, records, feature_rows, class_names, mode, limit=24):
    categories = np.asarray([category_of(record) for record in records], dtype=object)
    hard_idx = np.where(categories == "Hard")[0]
    if len(hard_idx) == 0:
        return []

    scaled = StandardScaler().fit_transform(x_values)
    if mode == "feature_extreme":
        feature_indices = [row["index"] for row in feature_rows[:5]]
        scores = np.max(np.abs(scaled[:, feature_indices]), axis=1)
        chosen = hard_idx[np.argsort(scores[hard_idx])[::-1][:limit]]
    elif mode == "feature_normal":
        distances = np.linalg.norm(scaled, axis=1)
        chosen = hard_idx[np.argsort(distances[hard_idx])[:limit]]
    else:
        raise ValueError(mode)

    result = []
    for index in chosen:
        record = dict(records[int(index)])
        label = int(record.get("label", -1))
        record["class_name"] = class_names[label] if 0 <= label < len(class_names) else str(label)
        result.append(record)
    return result


def main():
    ensure_dirs()
    records, lightweight_x, _, _ = build_dataset("lightweight_lgbm", max_samples=None)
    dataset = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=False, download=True, transform=None)
    class_names = dataset.classes
    for record in records:
        if "label" not in record:
            _, label = dataset[int(record["index"])]
            record["label"] = int(label)

    feature_sets = load_allowed_feature_space(records, lightweight_x)
    allowed_x, allowed_names = feature_sets["allowed_full"]
    feature_rows = feature_diagnostics(allowed_x, allowed_names, records)
    class_rows = class_diagnostics(records, class_names)
    collision_pairs, collision_summary = nearest_collision_pairs(allowed_x, records, class_names)

    artifact_dir = ARTIFACT_DIR / "difficulty_mechanism_decomposition"
    extreme_records = representative_hard_records(allowed_x, records, feature_rows, class_names, "feature_extreme")
    normal_records = representative_hard_records(allowed_x, records, feature_rows, class_names, "feature_normal")
    collision_grid_records = []
    for pair in collision_pairs[:8]:
        collision_grid_records.extend([pair["hard_record"], pair["easy_record"]])

    grids = {
        "hard_feature_extreme_grid": write_image_grid(
            artifact_dir / "hard_feature_extreme_grid.png",
            dataset,
            extreme_records,
            "Hard samples with extreme zero-latency feature values",
        ),
        "hard_feature_normal_grid": write_image_grid(
            artifact_dir / "hard_feature_normal_grid.png",
            dataset,
            normal_records,
            "Hard samples near the center of zero-latency feature space",
        ),
        "hard_easy_collision_pairs_grid": write_image_grid(
            artifact_dir / "hard_easy_collision_pairs_grid.png",
            dataset,
            collision_grid_records,
            "Nearest Hard/Easy pairs in allowed feature space",
            columns=4,
        ),
    }

    counts = category_counts(records)
    total = len(records)
    low_accuracy = 100.0 * (counts["Easy"] + counts["Inverse"]) / total
    high_accuracy = 100.0 * (counts["Easy"] + counts["Hard"]) / total

    summary = {
        "status": "ok",
        "purpose": "Mechanistic feasibility check: determine whether LOW/HIGH disagreement is observable in zero-latency image statistics.",
        "samples": total,
        "category_counts": counts,
        "low_accuracy": low_accuracy,
        "high_accuracy": high_accuracy,
        "feature_sets": {
            name: {"feature_count": int(values.shape[1])}
            for name, (values, _) in feature_sets.items()
        },
        "top_features_for_hard_vs_rest": feature_rows,
        "top_classes_by_hard_count": class_rows[:20],
        "top_classes_by_hard_rate_min_20_samples": [
            row for row in sorted(
                [row for row in class_rows if row["total"] >= 20],
                key=lambda item: item["hard_rate"],
                reverse=True,
            )[:20]
        ],
        "nearest_hard_easy_collision_summary": collision_summary,
        "nearest_hard_easy_collision_pairs": [
            {key: value for key, value in pair.items() if key not in {"hard_record", "easy_record"}}
            for pair in collision_pairs
        ],
        "representative_artifacts": grids,
        "interpretation_hooks": {
            "supports_low_level_observable_hypothesis": [
                "Hard samples concentrate in feature extremes.",
                "Top zero-latency features show high AUC or mutual information.",
                "Nearest Hard/Easy collisions are rare or far apart.",
            ],
            "supports_semantic_or_model_internal_hypothesis": [
                "Hard samples appear near the feature-space center.",
                "Hard classes are concentrated by class while low-level feature separation is weak.",
                "Nearest Hard/Easy collisions are close and visually/statistically similar.",
            ],
        },
    }

    output_path = RESULTS_DIR / "difficulty_mechanism_decomposition.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
