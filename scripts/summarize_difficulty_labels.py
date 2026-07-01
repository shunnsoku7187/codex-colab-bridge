import json
from collections import Counter

from src.experiment_paths import DIFFICULTY_LABELS_PATH


def main():
    data = json.loads(DIFFICULTY_LABELS_PATH.read_text(encoding="utf-8"))
    categories = Counter(item["category"] for item in data)
    low_acc = sum(item["low_correct"] for item in data) / len(data) * 100
    high_acc = sum(item["high_correct"] for item in data) / len(data) * 100
    print(f"samples: {len(data)}")
    print(f"low_accuracy: {low_acc:.2f}%")
    print(f"high_accuracy: {high_acc:.2f}%")
    for key, value in categories.items():
        print(f"{key}: {value} ({value / len(data) * 100:.1f}%)")


if __name__ == "__main__":
    main()
