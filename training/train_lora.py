"""LoRA Fine-Tuning for ASR Correction Model.

Extracted from FYP/Tests/train_lora.py for use as a library function.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingConfig:
    """Training hyperparameters and paths."""

    model_name: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    data_dir: Path = Path("./data/collected_data")
    train_file: str = "train.jsonl"
    valid_file: str = "valid.jsonl"
    lora_rank: int = 16
    lora_scale: float = 32.0
    lora_dropout: float = 0.05
    lora_layers: int = 24
    lora_keys: list = None
    batch_size: int = 4
    learning_rate: float = 2e-5
    iterations: int = 2000
    grad_checkpoint: bool = True
    steps_per_report: int = 10
    steps_per_eval: int = 100
    val_batches: int = -1
    adapter_path: Path = Path("./asr_correction/adapters")

    def __init__(self, **kwargs):
        self.lora_keys = ["self_attn.q_proj", "self_attn.v_proj"]
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


def train_with_mlx(config: TrainingConfig) -> dict:
    """Run LoRA fine-tuning using MLX. Returns result dict."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner import train as lora_train
    from mlx_lm.tuner.datasets import load_dataset
    from mlx_lm.tuner.trainer import TrainingArgs

    # Count data
    train_path = config.data_dir / config.train_file
    valid_path = config.data_dir / config.valid_file
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Validation data not found: {valid_path}")

    with open(train_path) as f:
        train_count = sum(1 for _ in f)
    with open(valid_path) as f:
        valid_count = sum(1 for _ in f)

    logger.info("Training: %d examples, Validation: %d examples", train_count, valid_count)

    # Load model
    logger.info("Loading model: %s", config.model_name)
    model, tokenizer = load(config.model_name)

    # Load datasets — mlx-lm expects a namespace with .data attribute
    from types import SimpleNamespace
    dataset_args = SimpleNamespace(
        data=str(config.data_dir),
        hf_dataset=False,
        train=True,
        test=False,
    )
    train_set, valid_set, _ = load_dataset(dataset_args, tokenizer)

    # Training args (mlx-lm current API)
    adapter_file = str(config.adapter_path / "adapters.safetensors")
    training_args = TrainingArgs(
        batch_size=config.batch_size,
        iters=config.iterations,
        steps_per_report=config.steps_per_report,
        steps_per_eval=config.steps_per_eval,
        steps_per_save=config.iterations,  # Save at end
        adapter_file=adapter_file,
        grad_checkpoint=config.grad_checkpoint,
    )

    lora_config = {
        "lora_parameters": {
            "rank": config.lora_rank,
            "scale": config.lora_scale,
            "dropout": config.lora_dropout,
            "keys": config.lora_keys,
        }
    }

    # Apply LoRA layers to model
    from mlx_lm.tuner.utils import linear_to_lora_layers
    linear_to_lora_layers(model, config.lora_layers, lora_config["lora_parameters"])

    # Create optimizer
    import mlx.optimizers as optim
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    # Train
    config.adapter_path.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    lora_train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=valid_set,
        args=training_args,
    )
    duration = time.time() - start_time

    # Save metadata
    metadata = {
        "completed_at": datetime.now().isoformat(),
        "duration_seconds": round(duration, 2),
        "config": {
            "model": config.model_name,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "iterations": config.iterations,
            "lora_rank": config.lora_rank,
        },
        "data": {
            "train_examples": train_count,
            "valid_examples": valid_count,
        },
    }
    metadata_path = config.adapter_path / "training_metadata.json"
    config.adapter_path.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata
