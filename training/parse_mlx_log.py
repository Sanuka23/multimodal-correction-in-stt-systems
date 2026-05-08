"""Parse mlx-lm training stdout into a JSON metrics file.

Usage:
    python training/parse_mlx_log.py logs/qwen35_v1.log > logs/qwen35_v1_metrics.json

The notebook training/finetune_qwen35.ipynb reads the produced JSON to render
loss / throughput / memory figures without hardcoded numbers.
"""

import json
import re
import sys
from pathlib import Path


# mlx-lm reports lines like:
#   Iter 10: Train loss 2.971, Learning Rate 1.000e-05, ...,
#                It/sec 0.470, Tokens/sec 170.5, Trained Tokens 4096, ...,
#                Peak mem 19.05 GB
RE_TRAIN = re.compile(
    r"Iter\s+(\d+):\s+Train loss\s+([\d.]+).*?Tokens/sec\s+([\d.]+).*?Peak mem\s+([\d.]+)\s*GB",
    re.IGNORECASE | re.DOTALL,
)

# Validation lines look like:
#   Iter 200: Val loss 1.547, Val took 23.4s
RE_VAL = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)", re.IGNORECASE)


def parse(text: str) -> dict:
    train, val, throughput, memory = [], [], [], []
    for line in text.splitlines():
        m = RE_TRAIN.search(line)
        if m:
            it = int(m.group(1))
            train.append({"iter": it, "loss": float(m.group(2))})
            throughput.append({"iter": it, "tokens_per_sec": float(m.group(3))})
            memory.append({"iter": it, "peak_gb": float(m.group(4))})
            continue
        m = RE_VAL.search(line)
        if m:
            val.append({"iter": int(m.group(1)), "loss": float(m.group(2))})

    return {
        "train": train,
        "val": val,
        "throughput": throughput,
        "memory": memory,
    }


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)
    metrics = parse(log_path.read_text(encoding="utf-8", errors="replace"))
    json.dump(metrics, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
