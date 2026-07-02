import argparse
import json
import math
from collections import Counter, defaultdict

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from scripts.difficulty_mechanism_decomposition import category_of
from scripts.prepare_difficulty_labels import load_high_model, load_low_model
from src.experiment_paths import ARTIFACT_DIR, DATA_DIR, DIFFICULTY_LABELS_PATH, RESULTS_DIR, ensure_dirs


def load_records(max_samples):
    records = json.loads(DIFFICULTY_LABELS_PATH.read_text(encoding="utf-8"))
    return records[:max_samples] if max_samples else records


def tensor_summaries(tensor):
    x = tensor.detach().float().cpu()
    if x.ndim == 4:
        flat = x.flatten(1)
        spatial = x.flatten(2)
        result = {
            "mean": x.mean(dim=(1, 2, 3)),
            "std": x.std(dim=(1, 2, 3), unbiased=False),
            "abs_mean": x.abs().mean(dim=(1, 2, 3)),
            "l2": torch.linalg.vector_norm(flat, dim=1),
            "max": flat.max(dim=1).values,
            "positive_frac": (flat > 0).float().mean(dim=1),
            "channel_mean_std": x.mean(dim=(2, 3)).std(dim=1, unbiased=False),
            "channel_energy_entropy": entropy_from_nonnegative(spatial.square().mean(dim=2)),
        }
    elif x.ndim == 3:
        cls = x[:, 0, :]
        flat = x.flatten(1)
        result = {
            "mean": x.mean(dim=(1, 2)),
            "std": x.std(dim=(1, 2), unbiased=False),
            "abs_mean": x.abs().mean(dim=(1, 2)),
            "l2": torch.linalg.vector_norm(flat, dim=1),
            "max": flat.max(dim=1).values,
            "cls_l2": torch.linalg.vector_norm(cls, dim=1),
            "token_mean_std": x.mean(dim=2).std(dim=1, unbiased=False),
        }
    elif x.ndim == 2:
        result = {
            "mean": x.mean(dim=1),
            "std": x.std(dim=1, unbiased=False),
            "abs_mean": x.abs().mean(dim=1),
            "l2": torch.linalg.vector_norm(x, dim=1),
            "max": x.max(dim=1).values,
            "positive_frac": (x > 0).float().mean(dim=1),
        }
    else:
        flat = x.reshape(x.shape[0], -1)
        result = {
            "mean": flat.mean(dim=1),
            "std": flat.std(dim=1, unbiased=False),
            "abs_mean": flat.abs().mean(dim=1),
            "l2": torch.linalg.vector_norm(flat, dim=1),
            "max": flat.max(dim=1).values,
        }
    return {
        key: values.numpy().astype(float).tolist()
        for key, values in result.items()
    }


def entropy_from_nonnegative(values):
    probs = values / (values.sum(dim=1, keepdim=True) + 1e-12)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
    return entropy / math.log(max(2, values.shape[1]))


def logits_summary(logits, labels, class_names, topk=5):
    probs = torch.softmax(logits, dim=1)
    top_probs, top_indices = torch.topk(probs, k=topk, dim=1)
    sorted_logits, sorted_indices = torch.sort(logits, dim=1, descending=True)
    pred = sorted_indices[:, 0]
    margins = sorted_logits[:, 0] - sorted_logits[:, 1]
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1) / math.log(probs.shape[1])
    rows = []
    for i in range(logits.shape[0]):
        label = int(labels[i].item())
        rank = int((sorted_indices[i] == label).nonzero(as_tuple=False)[0].item()) + 1
        top = []
        for cls, prob in zip(top_indices[i].tolist(), top_probs[i].tolist()):
            top.append({
                "class_index": int(cls),
                "class_name": class_names[int(cls)],
                "prob": float(prob),
            })
        rows.append({
            "pred": int(pred[i].item()),
            "pred_name": class_names[int(pred[i].item())],
            "correct": bool(pred[i].item() == label),
            "true_rank": rank,
            "true_prob": float(probs[i, label].item()),
            "true_logit": float(logits[i, label].item()),
            "pred_logit": float(logits[i, pred[i]].item()),
            "top1_prob": float(top_probs[i, 0].item()),
            "top2_prob": float(top_probs[i, 1].item()),
            "top1_top2_logit_margin": float(margins[i].item()),
            "entropy_norm": float(entropy[i].item()),
            "top5": top,
        })
    return rows


def choose_low_hook_modules(low_model, max_layers):
    if not hasattr(low_model, "features"):
        return []
    children = list(low_model.features.named_children())
    if not children:
        return []
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


def confusion_key(record):
    return f"{record['low_pred_name']} -> {record['label_name']}"


def aggregate(rows):
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    category_summary = {}
    for category, items in by_category.items():
        category_summary[category] = {
            "count": len(items),
            "low_true_rank_mean": float(np.mean([item["low"]["true_rank"] for item in items])),
            "high_true_rank_mean": float(np.mean([item["high"]["true_rank"] for item in items])),
            "low_margin_mean": float(np.mean([item["low"]["top1_top2_logit_margin"] for item in items])),
            "high_margin_mean": float(np.mean([item["high"]["top1_top2_logit_margin"] for item in items])),
            "low_entropy_mean": float(np.mean([item["low"]["entropy_norm"] for item in items])),
            "high_entropy_mean": float(np.mean([item["high"]["entropy_norm"] for item in items])),
        }

    hard_rows = [row for row in rows if row["category"] == "Hard"]
    hard_confusions = Counter(confusion_key(row) for row in hard_rows)
    return {
        "category_summary": category_summary,
        "top_hard_low_confusions": [
            {"confusion": key, "count": count}
            for key, count in hard_confusions.most_common(30)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--low-hook-layers", type=int, default=6)
    parser.add_argument("--high-hidden-layers", default="0,3,6,9,12")
    parser.add_argument("--output-prefix", default="model_process_trace_001")
    args = parser.parse_args()

    ensure_dirs()
    records = load_records(args.max_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} samples={len(records)}", flush=True)

    dataset = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=False, download=True, transform=None)
    class_names = dataset.classes
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

    low_hook_outputs = {}
    handles = []
    low_hook_modules = choose_low_hook_modules(low_model, args.low_hook_layers)
    for name, module in low_hook_modules:
        def make_hook(layer_name):
            def hook(_module, _inputs, output):
                low_hook_outputs[layer_name] = output.detach()
            return hook
        handles.append(module.register_forward_hook(make_hook(name)))
    print(f"LOW hook layers: {[name for name, _ in low_hook_modules]}", flush=True)

    requested_high_layers = [int(item.strip()) for item in args.high_hidden_layers.split(",") if item.strip()]
    trace_dir = ARTIFACT_DIR / "model_process_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    rows_path = trace_dir / f"{args.output_prefix}_rows.jsonl"
    summary_path = RESULTS_DIR / f"{args.output_prefix}_summary.json"

    rows = []
    low_model.eval()
    high_model.eval()
    with rows_path.open("w", encoding="utf-8") as writer:
        with torch.no_grad():
            for start in tqdm(range(0, len(records), args.batch_size), desc="Tracing model process"):
                batch_records = records[start:start + args.batch_size]
                images = [dataset[int(record["index"])][0] for record in batch_records]
                labels = torch.tensor([int(record["label"]) for record in batch_records], device=device)

                low_hook_outputs.clear()
                batch_low = torch.stack([transform_low(image) for image in images]).to(device)
                low_logits = low_model(batch_low)
                low_rows = logits_summary(low_logits.detach().cpu(), labels.detach().cpu(), class_names)
                low_layer_summaries = {
                    name: tensor_summaries(output)
                    for name, output in low_hook_outputs.items()
                }

                batch_high = torch.stack([transform_high(image) for image in images]).to(device)
                high_output = high_model(batch_high, output_hidden_states=True)
                high_logits = high_output.logits
                high_rows = logits_summary(high_logits.detach().cpu(), labels.detach().cpu(), class_names)
                high_layer_summaries = {}
                hidden_states = getattr(high_output, "hidden_states", None)
                if hidden_states is not None:
                    for layer_idx in requested_high_layers:
                        if 0 <= layer_idx < len(hidden_states):
                            high_layer_summaries[f"hidden_{layer_idx}"] = tensor_summaries(hidden_states[layer_idx])

                for i, record in enumerate(batch_records):
                    label = int(record["label"])
                    row = {
                        "index": int(record["index"]),
                        "label": label,
                        "label_name": class_names[label],
                        "category": record.get("category", category_of(record)),
                        "low": low_rows[i],
                        "high": high_rows[i],
                        "low_pred_name": low_rows[i]["pred_name"],
                        "high_pred_name": high_rows[i]["pred_name"],
                        "low_layers": {
                            layer_name: {
                                metric: float(values[i])
                                for metric, values in layer_values.items()
                            }
                            for layer_name, layer_values in low_layer_summaries.items()
                        },
                        "high_layers": {
                            layer_name: {
                                metric: float(values[i])
                                for metric, values in layer_values.items()
                            }
                            for layer_name, layer_values in high_layer_summaries.items()
                        },
                    }
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.append(row)

    for handle in handles:
        handle.remove()

    summary = {
        "status": "ok",
        "purpose": "Broad process trace for LOW/HIGH decisions across all difficulty categories.",
        "samples": len(rows),
        "device": str(device),
        "rows_jsonl": str(rows_path),
        "low_hook_layers": [name for name, _ in low_hook_modules],
        "high_hidden_layers_requested": requested_high_layers,
        "notes": [
            "This is an analysis artifact, not a runtime-claimable router feature set.",
            "Use these traces to identify where LOW begins to diverge from HIGH and which errors are confidence/margin/attention-like failures.",
        ],
        **aggregate(rows),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
