import argparse
import json
import math
from collections import Counter, defaultdict

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from scripts.prepare_difficulty_labels import load_high_model, load_low_model
from src.experiment_paths import ARTIFACT_DIR, DATA_DIR, DIFFICULTY_LABELS_PATH, RESULTS_DIR, ensure_dirs


RETENTION_TARGETS = [0.995, 0.99, 0.98, 0.95]
DEFAULT_HIGH_LAYERS = [0, 3, 6, 9, 12]
DEFAULT_LOW_LAYER_COUNT = 6


def load_records(max_samples):
    records = json.loads(DIFFICULTY_LABELS_PATH.read_text(encoding="utf-8"))
    return records[:max_samples] if max_samples else records


def category_of(record):
    if "category" in record:
        return record["category"]
    low = bool(record["low_correct"])
    high = bool(record["high_correct"])
    if low and high:
        return "Easy"
    if not low and high:
        return "Hard"
    if not low and not high:
        return "Impossible"
    return "Inverse"


def category_counts(categories):
    return dict(sorted(Counter(categories).items()))


def entropy_from_logits(logits):
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs + 1e-12)).sum(dim=1) / math.log(probs.shape[1])


def output_features(logits, prefix):
    probs = torch.softmax(logits, dim=1)
    top_probs, _top_indices = torch.topk(probs, k=2, dim=1)
    sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
    margin = sorted_logits[:, 0] - sorted_logits[:, 1]
    entropy = entropy_from_logits(logits)
    arr = torch.stack([
        top_probs[:, 0],
        top_probs[:, 1],
        margin,
        entropy,
    ], dim=1).detach().float().cpu().numpy()
    names = [
        f"{prefix}.top1_prob",
        f"{prefix}.top2_prob",
        f"{prefix}.top1_top2_logit_margin",
        f"{prefix}.entropy_norm",
    ]
    return arr.astype(np.float32), names


def tensor_summary_features(tensor, prefix):
    x = tensor.detach().float()
    if x.ndim == 4:
        flat = x.flatten(1)
        channel_mean = x.mean(dim=(2, 3))
        features = [
            x.mean(dim=(1, 2, 3)),
            x.std(dim=(1, 2, 3), unbiased=False),
            x.abs().mean(dim=(1, 2, 3)),
            torch.linalg.vector_norm(flat, dim=1),
            flat.max(dim=1).values,
            (flat > 0).float().mean(dim=1),
            channel_mean.std(dim=1, unbiased=False),
        ]
        names = ["mean", "std", "abs_mean", "l2", "max", "positive_frac", "channel_mean_std"]
    elif x.ndim == 3:
        cls = x[:, 0, :]
        flat = x.flatten(1)
        token_mean = x.mean(dim=2)
        features = [
            x.mean(dim=(1, 2)),
            x.std(dim=(1, 2), unbiased=False),
            x.abs().mean(dim=(1, 2)),
            torch.linalg.vector_norm(flat, dim=1),
            flat.max(dim=1).values,
            torch.linalg.vector_norm(cls, dim=1),
            token_mean.std(dim=1, unbiased=False),
        ]
        names = ["mean", "std", "abs_mean", "l2", "max", "cls_l2", "token_mean_std"]
    elif x.ndim == 2:
        features = [
            x.mean(dim=1),
            x.std(dim=1, unbiased=False),
            x.abs().mean(dim=1),
            torch.linalg.vector_norm(x, dim=1),
            x.max(dim=1).values,
            (x > 0).float().mean(dim=1),
        ]
        names = ["mean", "std", "abs_mean", "l2", "max", "positive_frac"]
    else:
        flat = x.reshape(x.shape[0], -1)
        features = [
            flat.mean(dim=1),
            flat.std(dim=1, unbiased=False),
            flat.abs().mean(dim=1),
            torch.linalg.vector_norm(flat, dim=1),
            flat.max(dim=1).values,
        ]
        names = ["mean", "std", "abs_mean", "l2", "max"]
    arr = torch.stack(features, dim=1).detach().cpu().numpy().astype(np.float32)
    return arr, [f"{prefix}.{name}" for name in names]


def choose_low_hook_modules(low_model, max_layers):
    if not hasattr(low_model, "features"):
        return []
    children = list(low_model.features.named_children())
    positions = np.linspace(0, len(children) - 1, num=min(max_layers, len(children)), dtype=int)
    result = []
    seen = set()
    for pos in positions.tolist():
        name, module = children[pos]
        full_name = f"features.{name}"
        if full_name not in seen:
            result.append((full_name, module))
            seen.add(full_name)
    return result


def build_feature_cache(records, cache_npz, metadata_path, batch_size, high_layers, low_layer_count, require_cuda):
    if cache_npz.exists() and metadata_path.exists():
        print(f"Using cached broad MID features: {cache_npz}", flush=True)
        loaded = np.load(cache_npz, allow_pickle=True)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return loaded, metadata

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} samples={len(records)}", flush=True)
    if require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required. In Colab, select a GPU runtime before running.")

    dataset = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=False, download=True, transform=None)
    transform_low = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])
    transform_high = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print("Loading LOW model...", flush=True)
    low_model = load_low_model(device)
    print("Loading HIGH model...", flush=True)
    high_model = load_high_model(device)
    low_model.eval()
    high_model.eval()

    low_hook_outputs = {}
    handles = []
    low_modules = choose_low_hook_modules(low_model, low_layer_count)
    for layer_name, module in low_modules:
        def make_hook(name):
            def hook(_module, _inputs, output):
                low_hook_outputs[name] = output.detach()
            return hook
        handles.append(module.register_forward_hook(make_hook(layer_name)))
    print(f"LOW hook layers: {[name for name, _ in low_modules]}", flush=True)
    print(f"HIGH hidden layers: {high_layers}", flush=True)

    buckets = defaultdict(list)
    feature_names = defaultdict(list)
    labels = []
    indices = []
    categories = []
    low_correct = []
    high_correct = []

    with torch.no_grad():
        for start in tqdm(range(0, len(records), batch_size), desc="Extracting broad MID features"):
            batch_records = records[start:start + batch_size]
            images = [dataset[int(record["index"])][0] for record in batch_records]

            low_hook_outputs.clear()
            low_batch = torch.stack([transform_low(image) for image in images]).to(device)
            low_logits = low_model(low_batch)
            arr, names = output_features(low_logits, "low_output")
            buckets["low_output_runtime"].append(arr)
            feature_names["low_output_runtime"] = names

            for layer_name, output in low_hook_outputs.items():
                arr, names = tensor_summary_features(output, f"low_layer.{layer_name}")
                buckets[f"low_layer:{layer_name}"].append(arr)
                feature_names[f"low_layer:{layer_name}"] = names

            high_batch = torch.stack([transform_high(image) for image in images]).to(device)
            high_output = high_model(high_batch, output_hidden_states=True)
            high_logits = high_output.logits
            arr, names = output_features(high_logits, "high_output")
            buckets["high_output_runtime"].append(arr)
            feature_names["high_output_runtime"] = names

            hidden_states = getattr(high_output, "hidden_states", None)
            if hidden_states is not None:
                for layer_idx in high_layers:
                    if 0 <= layer_idx < len(hidden_states):
                        arr, names = tensor_summary_features(hidden_states[layer_idx], f"high_hidden.{layer_idx}")
                        buckets[f"high_layer:hidden_{layer_idx}"].append(arr)
                        feature_names[f"high_layer:hidden_{layer_idx}"] = names

            for record in batch_records:
                indices.append(int(record["index"]))
                labels.append(int(record["label"]))
                categories.append(category_of(record))
                low_correct.append(bool(record["low_correct"]))
                high_correct.append(bool(record["high_correct"]))

    for handle in handles:
        handle.remove()

    arrays = {
        "index": np.asarray(indices, dtype=np.int64),
        "label": np.asarray(labels, dtype=np.int64),
        "category": np.asarray(categories),
        "low_correct": np.asarray(low_correct, dtype=bool),
        "high_correct": np.asarray(high_correct, dtype=bool),
    }
    for key, chunks in buckets.items():
        arrays[key] = np.concatenate(chunks, axis=0).astype(np.float32)

    metadata = {
        "samples": len(records),
        "feature_groups": sorted(buckets.keys()),
        "feature_names": {key: list(names) for key, names in feature_names.items()},
        "high_layers": high_layers,
        "low_layers": [name for name, _ in low_modules],
    }
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_npz, **arrays)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote feature cache: {cache_npz}", flush=True)
    loaded = np.load(cache_npz, allow_pickle=True)
    return loaded, metadata


def make_groups(data, metadata):
    groups = []
    available = set(metadata["feature_groups"])
    for key in sorted(available):
        if key.startswith("low_layer:"):
            groups.append({"name": key, "keys": [key], "cost_class": "during_low"})
        elif key.startswith("high_layer:"):
            groups.append({"name": key, "keys": [key], "cost_class": "partial_high"})
    base = [
        ("low_output_runtime", ["low_output_runtime"], "after_low"),
        ("high_output_runtime", ["high_output_runtime"], "full_high_output_diagnostic"),
    ]
    for name, keys, cost in base:
        if all(key in available for key in keys):
            groups.append({"name": name, "keys": keys, "cost_class": cost})

    low_layer_keys = sorted(key for key in available if key.startswith("low_layer:"))
    high_layer_keys = sorted(key for key in available if key.startswith("high_layer:"))
    if low_layer_keys:
        groups.append({"name": "all_low_layers", "keys": low_layer_keys, "cost_class": "during_low"})
    if high_layer_keys:
        groups.append({"name": "all_high_layers", "keys": high_layer_keys, "cost_class": "partial_or_full_high"})
    if "low_output_runtime" in available and low_layer_keys:
        groups.append({"name": "low_output_plus_low_layers", "keys": ["low_output_runtime", *low_layer_keys], "cost_class": "after_low"})
    if "low_output_runtime" in available and high_layer_keys:
        early_high = high_layer_keys[:2]
        groups.append({"name": "low_output_plus_early_high_layers", "keys": ["low_output_runtime", *early_high], "cost_class": "after_low_plus_partial_high"})
    if "low_output_runtime" in available and "high_output_runtime" in available:
        groups.append({"name": "low_plus_high_outputs_diagnostic", "keys": ["low_output_runtime", "high_output_runtime"], "cost_class": "full_high_output_diagnostic"})
    return groups


def concat_features(data, keys):
    return np.concatenate([np.asarray(data[key], dtype=np.float32) for key in keys], axis=1)


def oof_scores(x_values, labels, strata, seed):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    models = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=7,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }
    result = {}
    for model_name, model in models.items():
        scores = np.zeros(len(labels), dtype=np.float32)
        for train_idx, eval_idx in splitter.split(x_values, strata):
            model.fit(x_values[train_idx], labels[train_idx])
            scores[eval_idx] = model.predict_proba(x_values[eval_idx])[:, 1]
        result[model_name] = scores
    return result


def frontier(scores, categories, low_correct, high_correct):
    hard = categories == "Hard"
    easy = categories == "Easy"
    inverse = categories == "Inverse"
    impossible = categories == "Impossible"
    thresholds = np.unique(scores)
    candidates = []
    low_correct_total = int(low_correct.sum())
    hard_total = int(hard.sum())
    impossible_total = int(impossible.sum())
    for threshold in thresholds:
        selected = scores >= threshold
        if not selected.any():
            continue
        low_correct_selected = int((selected & low_correct).sum())
        hard_selected = int((selected & hard).sum())
        impossible_selected = int((selected & impossible).sum())
        row = {
            "threshold": float(threshold),
            "selected": int(selected.sum()),
            "selected_rate": float(selected.mean()),
            "low_correct_retention": float(1.0 - low_correct_selected / low_correct_total),
            "low_correct_selected": low_correct_selected,
            "hard_recovery": float(hard_selected / hard_total),
            "hard_selected": hard_selected,
            "impossible_selected_rate": float(impossible_selected / impossible_total),
            "selected_easy": int((selected & easy).sum()),
            "selected_hard": hard_selected,
            "selected_impossible": impossible_selected,
            "selected_inverse": int((selected & inverse).sum()),
            "selected_high_correct": int((selected & high_correct).sum()),
            "precision_hard_among_selected": float(hard_selected / int(selected.sum())),
        }
        candidates.append(row)
    by_retention = {}
    for target in RETENTION_TARGETS:
        feasible = [row for row in candidates if row["low_correct_retention"] >= target]
        by_retention[str(target)] = max(
            feasible,
            key=lambda row: (row["hard_recovery"], row["precision_hard_among_selected"], row["selected"]),
        ) if feasible else None
    return by_retention


def deadline_control_candidates():
    return [
        {
            "name": "remaining_time_rule",
            "description": "Run the deepest stage that still fits the remaining deadline.",
            "strength": "Easiest to explain and implement.",
            "risk": "May waste work on samples whose expected correction value is low.",
        },
        {
            "name": "safety_first_rule",
            "description": "If a reliable stage cannot finish, choose safe eject or re-sort.",
            "strength": "Directly avoids cascade-style deadline misses.",
            "risk": "Can become too conservative and lose useful decisions.",
        },
        {
            "name": "value_density_rule",
            "description": "Run MID/HIGH only when expected gain per added latency or energy is high.",
            "strength": "Best conceptual contrast against fixed cascade and always-parallel inference.",
            "risk": "Needs calibrated gain estimates; calibration may be fragile.",
        },
        {
            "name": "conservative_correction_rule",
            "description": "Keep LOW's answer unless the correction signal is very strong.",
            "strength": "Matches the 99% LOW-correct retention requirement.",
            "risk": "May recover too little Hard if correction evidence is weak.",
        },
    ]


def evaluate_groups(data, metadata, seed):
    categories = np.asarray(data["category"]).astype(str)
    labels = (categories == "Hard").astype(np.int64)
    low_correct = np.asarray(data["low_correct"], dtype=bool)
    high_correct = np.asarray(data["high_correct"], dtype=bool)
    strata = categories
    results = []
    for group in make_groups(data, metadata):
        x_values = concat_features(data, group["keys"])
        model_scores = oof_scores(x_values, labels, strata, seed)
        model_results = []
        for model_name, scores in model_scores.items():
            auc = float(roc_auc_score(labels, scores))
            by_retention = frontier(scores, categories, low_correct, high_correct)
            model_results.append({
                "model": model_name,
                "hard_auc": auc,
                "frontier_by_low_correct_retention": by_retention,
            })
        best_99 = None
        for model_result in model_results:
            row = model_result["frontier_by_low_correct_retention"].get("0.99")
            if row and (best_99 is None or row["hard_recovery"] > best_99["hard_recovery"]):
                best_99 = {
                    "model": model_result["model"],
                    "hard_auc": model_result["hard_auc"],
                    **row,
                }
        result = {
            **group,
            "feature_count": int(x_values.shape[1]),
            "model_results": model_results,
            "best_at_low_correct_retention_0.99": best_99,
        }
        results.append(result)
        print(json.dumps({
            "group": group["name"],
            "cost_class": group["cost_class"],
            "feature_count": int(x_values.shape[1]),
            "best_at_0.99": best_99,
        }, ensure_ascii=False), flush=True)
    return sorted(
        results,
        key=lambda item: -1 if item["best_at_low_correct_retention_0.99"] is None else item["best_at_low_correct_retention_0.99"]["hard_recovery"],
        reverse=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-name", default="deadline_aware_broad_feasibility_001.json")
    parser.add_argument("--cache-name", default="deadline_aware_broad_features_001")
    parser.add_argument("--high-hidden-layers", default="0,3,6,9,12")
    parser.add_argument("--low-hook-layers", type=int, default=DEFAULT_LOW_LAYER_COUNT)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    records = load_records(args.max_samples)
    high_layers = [int(item.strip()) for item in args.high_hidden_layers.split(",") if item.strip()]
    cache_npz = ARTIFACT_DIR / f"{args.cache_name}.npz"
    metadata_path = ARTIFACT_DIR / f"{args.cache_name}_metadata.json"
    data, metadata = build_feature_cache(
        records,
        cache_npz,
        metadata_path,
        args.batch_size,
        high_layers,
        args.low_hook_layers,
        args.require_cuda,
    )
    group_results = evaluate_groups(data, metadata, args.seed)
    categories = np.asarray(data["category"]).astype(str)
    low_correct = np.asarray(data["low_correct"], dtype=bool)
    high_correct = np.asarray(data["high_correct"], dtype=bool)
    summary = {
        "status": "ok",
        "purpose": "Broad feasibility check for a real conservative MID signal and deadline-aware control candidates.",
        "samples": int(len(categories)),
        "category_counts": category_counts(categories),
        "low_accuracy": float(low_correct.mean()),
        "high_accuracy": float(high_correct.mean()),
        "retention_targets": RETENTION_TARGETS,
        "feature_cache": str(cache_npz),
        "feature_metadata": metadata,
        "decision_rule": [
            "Promising: non-oracle group reaches LOW-correct retention >= 0.99 and Hard recovery >= 0.40.",
            "Strong: same condition holds for after_low or partial_high features rather than oracle/full-high-output diagnostics.",
            "Weak: only oracle/full-high-output groups pass, or Hard recovery collapses under 99% LOW-correct retention.",
        ],
        "deadline_control_candidates": deadline_control_candidates(),
        "group_results": group_results,
    }
    output_path = RESULTS_DIR / args.output_name
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
