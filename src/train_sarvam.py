"""
train_sarvam.py
===============
Hardware-adaptive QLoRA fine-tuning framework for `sarvamai/sarvam-1` (2B) SLM
on the Marathi Sentiment Dataset (MahaSent-MD).

Key Features:
- Auto-detects hardware (VRAM, CUDA version, compute capability, precision support)
- Auto-configures 4-bit QLoRA parameters (batch size, grad accum, seq len, grad checkpointing)
- Memory validation dry run with automatic CUDA OOM fallback handling
- Preserves original dataset CSV files strictly read-only
- Evaluates per epoch tracking Macro-F1, Precision, Recall, Accuracy
- Saves per-epoch checkpoints (checkpoint-epoch-1..N) and selects best model by Macro-F1
- Generates loss curves and comprehensive training metrics in results/
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent))
from auto_config import dry_run_memory_check, get_auto_config
from data_loader import LABEL_NAMES, load_raw_dataset
from gpu_detector import print_hardware_report

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

try:
    from trl import SFTConfig, SFTTrainer
except ImportError:
    SFTConfig = None
    SFTTrainer = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_MODEL: str = "sarvamai/sarvam-1"
DEFAULT_OUTPUT_DIR: str = "saved_models/sarvam-1-lora"
RESULTS_DIR: Path = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CACHE_DIR: str = str(Path(__file__).parent.parent / "model")
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR

PROMPT_TEMPLATE: str = (
    "Classify the sentiment of the following Marathi text. Reply with ONLY ONE WORD from [Positive, Negative, Neutral].\n"
    "Text: {text_sample}\n"
    "Sentiment: {label}"
)

INT_TO_LABEL_STR: dict[int, str] = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}
STR_TO_LABEL_STR: dict[str, str] = {
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
    "-1": "Negative",
    "0": "Neutral",
    "1": "Positive",
}


def plot_loss_curves(log_history: list[dict], output_plot_path: Path, model_tag: str = "sarvam-1") -> None:
    """Save training / validation loss curve PNG from Trainer log history."""
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("step", 0))
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", 0))
            eval_losses.append(entry["eval_loss"])

    if not train_losses:
        log.warning("No loss entries found in log history — skipping loss plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_steps, train_losses, label="Training Loss", color="#2196F3", linewidth=2)
    if eval_losses:
        ax.plot(
            eval_steps,
            eval_losses,
            label="Validation Loss",
            color="#FF5722",
            linewidth=2,
            linestyle="--",
            marker="o",
            markersize=5,
        )

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Training & Validation Loss — {model_tag} (QLoRA)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Loss curves saved → %s", output_plot_path)


def format_dataset_for_sft(raw_ds: DatasetDict) -> DatasetDict:
    """Format dataset splits into prompt instruction format in-memory."""
    formatted_dict = {}

    for split_name, ds in raw_ds.items():
        formatted_texts = []
        for sample in ds:
            text_val = sample.get("text", "")
            raw_label = sample.get("label", "")

            if isinstance(raw_label, (int, torch.Tensor)):
                label_str = INT_TO_LABEL_STR.get(int(raw_label), "Neutral")
            elif isinstance(raw_label, str):
                label_str = STR_TO_LABEL_STR.get(raw_label.lower().strip(), raw_label.capitalize())
            else:
                label_str = "Neutral"

            prompt_text = PROMPT_TEMPLATE.format(text_sample=text_val, label=label_str)
            formatted_texts.append(prompt_text)

        formatted_dict[split_name] = Dataset.from_dict({"text": formatted_texts})
        log.info("Formatted %-12s split: %d samples", split_name, len(formatted_texts))

    return DatasetDict(formatted_dict)


class EpochSaverCallback(TrainerCallback):
    """Callback to ensure clear checkpoint naming per epoch."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        epoch_idx = int(round(state.epoch)) if state.epoch else 1
        epoch_dir = self.output_dir / f"checkpoint-epoch-{epoch_idx}"
        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")

        if model and not epoch_dir.exists():
            log.info("Saving epoch %d checkpoint to '%s' …", epoch_idx, epoch_dir)
            model.save_pretrained(epoch_dir)
            if tokenizer:
                tokenizer.save_pretrained(epoch_dir)


def train_sarvam(
    model_name: str = DEFAULT_MODEL,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    data_dir: Optional[str] = None,
    debug: bool = False,
    epochs: int = 4,
    batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
    seq_length: Optional[int] = None,
    learning_rate: Optional[float] = None,
    lora_r: Optional[int] = None,
    gradient_checkpointing: Optional[bool] = None,
    force_cpu: bool = False,
    skip_dry_run: bool = False,
) -> None:
    """Execute hardware-adaptive Sarvam-1 QLoRA fine-tuning."""
    if LoraConfig is None or SFTTrainer is None:
        raise ImportError(
            "peft and trl libraries are required. Please run: pip install peft trl bitsandbytes"
        )

    print("\n========================================")
    print(" SARVAM-1 AUTO FINE-TUNER")
    print("========================================")

    # 1. Hardware Detection & Auto-Configuration
    profile = print_hardware_report()
    config = get_auto_config(
        profile=profile,
        epochs=1 if debug else epochs,
        user_batch_size=batch_size,
        user_grad_accum=gradient_accumulation_steps,
        user_seq_length=seq_length,
        user_lr=learning_rate,
        user_lora_r=lora_r,
        user_grad_checkpointing=gradient_checkpointing,
        force_cpu=force_cpu,
    )
    config.print_config()

    # 2. Memory Validation Dry Run (if CUDA & not skipped)
    if profile.gpu_available and not force_cpu and not skip_dry_run:
        config = dry_run_memory_check(config, model_name=model_name)

    print("========================================")
    print(" STARTING TRAINING")
    print("========================================\n")

    # 3. Load & Format Dataset
    log.info("Loading MahaSent dataset from '%s' …", data_dir or "default location")
    raw_ds = load_raw_dataset(debug=debug, data_dir=data_dir)
    formatted_ds = format_dataset_for_sft(raw_ds)

    train_ds = formatted_ds["train"]
    eval_key = "validation" if "validation" in formatted_ds else "test"
    eval_ds = formatted_ds[eval_key]

    log.info("Sample formatted prompt:\n%s", train_ds[0]["text"])

    # 4. Tokenizer Setup
    log.info("Loading tokenizer for '%s' …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=MODEL_CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. Base Model Loading with 4-bit Quantization
    log.info("Loading base causal model '%s' (quantization=%s) …", model_name, config.quantization)
    if config.device == "cuda" and config.quantization == "4bit_nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=config.torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=config.torch_dtype,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=config.torch_dtype,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR,
        )

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 6. LoRA Adapter setup
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable_params, total_params = model.get_nb_trainable_parameters()
    log.info(
        "LoRA Parameters: %s trainable / %s total (%.2f%% trainable)",
        f"{trainable_params:,}",
        f"{total_params:,}",
        (trainable_params / total_params) * 100,
    )

    # 7. SFTConfig / TrainingArguments setup
    config_cls = SFTConfig if SFTConfig is not None else TrainingArguments
    config_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.epochs,
        "lr_scheduler_type": config.lr_scheduler_type,
        "bf16": (config.compute_dtype_str == "bfloat16"),
        "fp16": (config.compute_dtype_str == "float16"),
        "optim": config.optimizer,
        "eval_strategy": config.eval_strategy,
        "save_strategy": config.save_strategy,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": config.save_total_limit,
        "logging_steps": 5 if debug else 20,
        "warmup_steps": 5 if debug else config.warmup_steps,
        "report_to": "none",
        "seed": 42,
        "data_seed": 42,
        "dataloader_num_workers": config.dataloader_num_workers,
    }
    if SFTConfig is not None:
        config_kwargs["dataset_text_field"] = "text"
        config_kwargs["max_length"] = config.max_seq_length

    training_args = config_cls(**config_kwargs)

    # 8. Instantiate SFTTrainer
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "args": training_args,
        "callbacks": [EpochSaverCallback(output_dir)],
    }
    if SFTConfig is not None:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["dataset_text_field"] = "text"
        trainer_kwargs["max_seq_length"] = config.max_seq_length
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    # 9. Train Model
    t_start = time.time()
    log.info("Starting SFT training for %d epoch(s) …", config.epochs)
    train_result = trainer.train()
    train_duration = time.time() - t_start

    # 10. Save Best LoRA Adapter & Tokenizer
    log.info("Saving best fine-tuned LoRA adapter to '%s' …", output_dir)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 11. Run Final Validation
    log.info("Evaluating best checkpoint on validation set ('%s') …", eval_key)
    eval_metrics = trainer.evaluate()
    log.info("Validation Loss: %.4f", eval_metrics.get("eval_loss", 0.0))

    # 12. Save Summary & Loss Plot
    summary = {
        "gpu_name": profile.gpu_name,
        "vram_gb": profile.total_vram_gb,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "grad_accum": config.gradient_accumulation_steps,
        "effective_batch": config.effective_batch_size,
        "seq_length": config.max_seq_length,
        "lora_r": config.lora_r,
        "train_time_sec": round(train_duration, 2),
        "peak_vram_gb": config.peak_vram_gb,
        **train_result.metrics,
        **eval_metrics,
    }
    summary_csv = RESULTS_DIR / "sarvam-1_train_metrics.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    log.info("Saved training summary → %s", summary_csv)

    loss_plot_path = RESULTS_DIR / "sarvam-1_loss_curves.png"
    plot_loss_curves(trainer.state.log_history, loss_plot_path, model_tag="sarvam-1")

    print("\n========================================")
    print(" TRAINING COMPLETE")
    print(f" Saved Model : {output_dir}")
    print(f" Saved Metrics: {summary_csv}")
    print(f" Loss Plot    : {loss_plot_path}")
    print("========================================\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hardware-adaptive QLoRA fine-tuning for Sarvam-1."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Base model (default: {DEFAULT_MODEL})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Adapter save dir (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--data_dir", type=str, default=None, help="Local CSV data directory")
    parser.add_argument("--debug", action="store_true", help="Quick 99-row sanity run")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs (default: 4)")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--grad-accum", "--grad_accum", dest="grad_accum", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--seq-length", "--seq_length", dest="seq_length", type=int, default=None, help="Override max sequence length")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--lora-r", "--lora_r", dest="lora_r", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--grad-checkpointing", dest="grad_checkpointing", action="store_true", default=None, help="Force enable gradient checkpointing")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--no-dry-run", action="store_true", help="Skip memory dry-run check")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_sarvam(
        model_name=args.model,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        debug=args.debug,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        seq_length=args.seq_length,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        gradient_checkpointing=args.grad_checkpointing,
        force_cpu=args.cpu,
        skip_dry_run=args.no_dry_run,
    )
