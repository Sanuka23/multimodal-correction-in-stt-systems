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

    model_name: str = "mlx-community/Qwen3.5-9B-MLX-4bit"
    data_dir: Path = Path("./data/collected_data")
    train_file: str = "train.jsonl"
    valid_file: str = "valid.jsonl"
    lora_rank: int = 16
    lora_scale: float = 16.0
    lora_dropout: float = 0.0
    lora_layers: int = 8
    lora_keys: list = None
    batch_size: int = 4
    learning_rate: float = 1e-5
    iterations: int = 2000
    grad_checkpoint: bool = True
    steps_per_report: int = 10
    steps_per_eval: int = 500
    val_batches: int = 25
    adapter_path: Path = Path("./asr_correction/adapters")

    def __init__(self, **kwargs):
        self.lora_keys = ["self_attn.q_proj", "self_attn.v_proj"]
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


def train_with_mlx(config: TrainingConfig) -> dict:
    """Run LoRA fine-tuning using MLX. Returns result dict."""
    import gc

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
    logger.info("Config: iterations=%d, batch_size=%d, lr=%s, lora_rank=%d, lora_layers=%d",
                config.iterations, config.batch_size, config.learning_rate,
                config.lora_rank, config.lora_layers)

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

    # Wrap ChatDataset to return tokenized items (needed by iterate_batches)
    class TokenizedDataset:
        def __init__(self, chat_dataset):
            self._ds = chat_dataset
        def __len__(self):
            return len(self._ds)
        def __getitem__(self, idx):
            return self._ds.process(self._ds._data[idx])

    train_set = TokenizedDataset(train_set)
    valid_set = TokenizedDataset(valid_set)

    # Save checkpoints periodically to avoid losing progress
    steps_per_save = min(500, config.iterations)

    # Training args (mlx-lm current API)
    adapter_file = str(config.adapter_path / "adapters.safetensors")
    training_args = TrainingArgs(
        batch_size=config.batch_size,
        iters=config.iterations,
        steps_per_report=config.steps_per_report,
        steps_per_eval=config.steps_per_eval,
        steps_per_save=steps_per_save,
        val_batches=config.val_batches,
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

    # Freeze everything, then unfreeze only LoRA params.
    # Without this, biases/layernorm weights (float16) remain trainable
    # and Adam optimizer updates on float16 cause NaN overflow.
    model.freeze()
    model.unfreeze(keys=["lora_a", "lora_b"])

    from mlx.utils import tree_flatten
    trainable = tree_flatten(model.trainable_parameters())
    logger.info("Trainable parameters: %d (all LoRA)", len(trainable))

    # Create optimizer
    import mlx.optimizers as optim
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    # Train
    config.adapter_path.mkdir(parents=True, exist_ok=True)
    logger.info("Starting training for %d iterations (saving every %d steps)...",
                config.iterations, steps_per_save)
    start_time = time.time()

    try:
        lora_train(
            model=model,
            optimizer=optimizer,
            train_dataset=train_set,
            val_dataset=valid_set,
            args=training_args,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user at iteration. Saving current state...")
    except Exception as e:
        logger.error("Training failed: %s", e)
        raise
    finally:
        # Always try to clean up memory
        gc.collect()

    duration = time.time() - start_time
    logger.info("Training finished in %.1f seconds (%.1f minutes)", duration, duration / 60)

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
            "lora_layers": config.lora_layers,
            "grad_checkpoint": config.grad_checkpoint,
        },
        "data": {
            "train_examples": train_count,
            "valid_examples": valid_count,
        },
    }
    metadata_path = config.adapter_path / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", metadata_path)

    return metadata
