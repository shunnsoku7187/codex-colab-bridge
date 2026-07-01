import json
import time

import torch
from torch import nn

from src.experiment_paths import RESULTS_DIR, ensure_dirs


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 10),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ensure_dirs()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Set Colab runtime to GPU and rerun the job.")

    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = TinyClassifier().to(device).eval()
    batch = torch.randn(64, 3, 32, 32, device=device)

    with torch.inference_mode():
        for _ in range(5):
            _ = model(batch)
        torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(batch)
        torch.cuda.synchronize()
        duration_ms = (time.perf_counter() - start) * 1000

    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1)
    result = {
        "status": "ok",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "batch_shape": list(batch.shape),
        "logits_shape": list(logits.shape),
        "duration_ms": round(duration_ms, 4),
        "first_10_predictions": pred[:10].detach().cpu().tolist(),
        "first_row_probability_sum": round(float(probs[0].sum().detach().cpu().item()), 6),
        "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3),
    }
    output_path = RESULTS_DIR / "gpu_inference_smoke_output.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
