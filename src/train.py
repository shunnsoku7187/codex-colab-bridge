import argparse
import json
from pathlib import Path


def parse_simple_yaml(path):
    data = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out", default="results/train_local_result.json")
    args = parser.parse_args()

    config = parse_simple_yaml(args.config)

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    except Exception as exc:
        cuda_available = False
        device = f"torch unavailable: {exc}"

    result = {
        "config": config,
        "cuda_available": cuda_available,
        "device": device,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
