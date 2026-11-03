"""
train_cls_lora.py
=================
Fine-tune Qwen/Qwen3.5-4B (or any decoder LM) with a LoRA adapter +
a trainable linear classification head (SEQ_CLS task) for 3-class
Marathi sentiment.

NOT generative — uses AutoModelForSequenceClassification + LoRA (SEQ_CLS).
No QLoRA, no 4-bit quantisation, no text generation.

Usage
-----
# Debug run (≈500 samples, 50 steps)
python src/train_cls_lora.py --model Qwen/Qwen3.5-4B --data_dir data/ --debug

# Full training
python src/train_cls_lora.py --model Qwen/Qwen3.5-4B --data_dir data/ \\
    --epochs 1 --batch_size 32 --grad_accum 1 --lr 1e-4 \\
    --lora_r 16 --lora_alpha 32 --max_length 128 --seed 42 \\
    --output_name qwen3.5-4b-cls-lora
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    LABEL_NAMES,
    TEXT_COLUMN,
    LABEL_COLUMN,
    load_raw_dataset,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_SLUG = "qwen3.5-4b-cls-lora"
RESULTS_DIR = Path("results")
SAVED_MODELS_DIR = Path("saved_models")
SEED = 42

# LoRA target modules for Qwen2/Qwen3 family
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# id2label / label2id (MUST match data_loader.LABEL_NAMES)
# LABEL_NAMES = ["negative", "neutral", "positive"]  → 0, 1, 2
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path, output_name: str) -> logging.Logger:
    """Configure root logger with console + file handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train_console.log"

    fmt = "%(asctime)s  [%(levelname)s]  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Remove any existing handlers so we start clean
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(fh)

    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def pick_gpu_and_log(log: logging.Logger) -> tuple[str, str, float]:
    """
    Select GPU 0 (the 24 GB RTX 4090), set CUDA_VISIBLE_DEVICES,
    and return (device_str, gpu_name, vram_gb).
    """
    if not torch.cuda.is_available():
        log.warning("No CUDA GPU found — will train on CPU (slow).")
        return "cpu", "CPU", 0.0

    # If the user overrode CUDA_VISIBLE_DEVICES, respect it; else default to 0
    cv = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = cv
    torch.cuda.set_device(0)

    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9

    log.info("=" * 60)
    log.info("GPU  : %s", gpu_name)
    log.info("VRAM : %.1f GB  (CUDA_VISIBLE_DEVICES=%s)", vram_gb, cv)
    log.info("=" * 60)
    return "cuda", gpu_name, vram_gb


def peak_vram_gb() -> float:
    """Return peak allocated VRAM in GB (0 if no CUDA)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(0) / 1e9


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune decoder LM + LoRA cls head on Marathi sentiment.")
    p.add_argument("--model",       type=str,   default="Qwen/Qwen3.5-4B")
    p.add_argument("--data_dir",    type=str,   default="data/")
    p.add_argument("--max_length",  type=int,   default=128)
    p.add_argument("--epochs",      type=int,   default=1)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--grad_accum",  type=int,   default=1)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--lora_r",      type=int,   default=16)
    p.add_argument("--lora_alpha",  type=int,   default=32)
    p.add_argument("--lora_dropout",type=float, default=0.05)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--debug",       action="store_true",
                   help="~500 samples, 50 steps — fast sanity-check.")
    p.add_argument("--output_name", type=str,   default=MODEL_SLUG)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro":        f1_score(labels, preds, average="macro",  zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro":    recall_score(labels, preds, average="macro",    zero_division=0),
        "accuracy":        accuracy_score(labels, preds),
        # per-class F1
        "f1_negative":     f1_score(labels, preds, labels=[0], average="macro", zero_division=0),
        "f1_neutral":      f1_score(labels, preds, labels=[1], average="macro", zero_division=0),
        "f1_positive":     f1_score(labels, preds, labels=[2], average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Live-logging callback
# ---------------------------------------------------------------------------

class LiveLogCallback(TrainerCallback):
    """
    Appends one row to train_steps.csv (on_log) and val_metrics.csv
    (on_evaluate) after every HF-Trainer log event.  Survives crashes because
    every flush is written to disk immediately.
    """

    def __init__(self, log_dir: Path, start_time: float) -> None:
        self.log_dir = log_dir
        self.start_time = start_time
        self._step_csv  = log_dir / "train_steps.csv"
        self._val_csv   = log_dir / "val_metrics.csv"
        self._step_header_written = False
        self._val_header_written  = False
        self._log = logging.getLogger(__name__)

    def _gpu_mem_gb(self) -> float:
        return torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0.0

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" not in logs and "eval_loss" not in logs:
            return
        if "eval_loss" in logs:
            return  # handled by on_evaluate

        elapsed = time.time() - self.start_time
        row = {
            "step":         state.global_step,
            "epoch":        round(state.epoch or 0.0, 4),
            "train_loss":   logs.get("loss", ""),
            "learning_rate":logs.get("learning_rate", ""),
            "grad_norm":    logs.get("grad_norm", ""),
            "elapsed_sec":  round(elapsed, 1),
            "gpu_mem_gb":   round(self._gpu_mem_gb(), 3),
        }
        self._append_csv(self._step_csv, row, "step")

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is None:
            return
        row = {
            "step":              state.global_step,
            "epoch":             round(state.epoch or 0.0, 4),
            "val_loss":          metrics.get("eval_loss", ""),
            "val_accuracy":      metrics.get("eval_accuracy", ""),
            "val_f1_macro":      metrics.get("eval_f1_macro", ""),
            "val_precision_macro": metrics.get("eval_precision_macro", ""),
            "val_recall_macro":  metrics.get("eval_recall_macro", ""),
            "f1_negative":       metrics.get("eval_f1_negative", ""),
            "f1_neutral":        metrics.get("eval_f1_neutral", ""),
            "f1_positive":       metrics.get("eval_f1_positive", ""),
        }
        self._append_csv(self._val_csv, row, "step")

    def _append_csv(self, path: Path, row: dict, key: str) -> None:
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_loss_curves(log_dir: Path, log_history: list[dict], output_name: str) -> None:
    train_steps, train_losses = [], []
    eval_steps,  eval_losses  = [], []
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("step", 0))
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", 0))
            eval_losses.append(entry["eval_loss"])

    if not train_losses:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_steps, train_losses, label="Training Loss",   color="#2196F3", linewidth=2)
    if eval_losses:
        ax.plot(eval_steps, eval_losses, label="Validation Loss", color="#FF5722",
                linewidth=2, linestyle="--", marker="o", markersize=5)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title(f"Training & Validation Loss — {output_name}", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()

    # Save to both results/ and the log dir
    out1 = RESULTS_DIR / f"{output_name}_loss_curves.png"
    out2 = log_dir / "loss_curves.png"
    for out in (out1, out2):
        plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_val_f1_accuracy(log_dir: Path, log_history: list[dict]) -> None:
    steps, f1s, accs = [], [], []
    for entry in log_history:
        if "eval_f1_macro" in entry:
            steps.append(entry.get("step", 0))
            f1s.append(entry["eval_f1_macro"])
            accs.append(entry.get("eval_accuracy", float("nan")))
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, f1s,  label="Val Macro F1",  color="#4CAF50", linewidth=2, marker="o", markersize=4)
    ax.plot(steps, accs, label="Val Accuracy",  color="#9C27B0", linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Step"); ax.set_ylabel("Score")
    ax.set_title("Validation F1 & Accuracy vs Step", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(log_dir / "val_f1_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_lr_schedule(log_dir: Path, log_history: list[dict]) -> None:
    steps, lrs = [], []
    for entry in log_history:
        if "learning_rate" in entry and "eval_loss" not in entry:
            steps.append(entry.get("step", 0))
            lrs.append(entry["learning_rate"])
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, lrs, color="#FF9800", linewidth=2)
    ax.set_xlabel("Step"); ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(log_dir / "lr_schedule.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=".", stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def save_system_info(log_dir: Path, gpu_name: str, vram_gb: float, args: argparse.Namespace) -> None:
    import transformers
    import peft
    info = {
        "gpu_name":            gpu_name,
        "vram_gb":             round(vram_gb, 2),
        "cuda_version":        torch.version.cuda or "N/A",
        "torch_version":       torch.__version__,
        "transformers_version":transformers.__version__,
        "peft_version":        peft.__version__,
        "os":                  platform.platform(),
        "seed":                args.seed,
        "git_commit":          get_git_commit(),
        "model":               args.model,
        "max_length":          args.max_length,
        "epochs":              args.epochs,
        "batch_size":          args.batch_size,
        "grad_accum":          args.grad_accum,
        "lr":                  args.lr,
        "lora_r":              args.lora_r,
        "lora_alpha":          args.lora_alpha,
    }
    with open(log_dir / "system_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Directories
    output_name = args.output_name
    model_save_dir = SAVED_MODELS_DIR / output_name
    log_dir = RESULTS_DIR / output_name / "logs"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(log_dir, output_name)
    log.info("=" * 60)
    log.info("train_cls_lora.py  —  %s", output_name)
    log.info("=" * 60)

    # Seed
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # GPU
    device, gpu_name, vram_gb = pick_gpu_and_log(log)
    save_system_info(log_dir, gpu_name, vram_gb, args)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    log.info("Loading dataset from '%s' …", args.data_dir)
    # In debug mode load ~500 samples (≥99 per split from data_loader, but
    # we'll use the data_loader DEBUG_N_ROWS logic for train only)
    # We want ~500 train and keep val/test at standard debug size (99).
    # Override: use debug=True gives 99; we want 500 so we do it manually.
    raw_ds = load_raw_dataset(debug=False, data_dir=args.data_dir)

    if args.debug:
        from data_loader import _stratified_sample, DEBUG_N_ROWS
        debug_n = 498  # divisible by 3 → 166 per class
        raw_ds["train"]      = _stratified_sample(raw_ds["train"],      debug_n)
        raw_ds["validation"] = _stratified_sample(raw_ds["validation"], DEBUG_N_ROWS)
        raw_ds["test"]       = _stratified_sample(raw_ds["test"],       DEBUG_N_ROWS)
        log.info("[DEBUG] Subsampled — train: %d, val: %d, test: %d",
                 len(raw_ds["train"]), len(raw_ds["validation"]), len(raw_ds["test"]))

    for split in raw_ds:
        ds = raw_ds[split]
        labels_arr = np.array(ds[LABEL_COLUMN])
        counts = {LABEL_NAMES[i]: int((labels_arr == i).sum()) for i in range(3)}
        log.info("  %-12s  %d rows  %s", split, len(ds), counts)

    # ── 2. Tokeniser ─────────────────────────────────────────────────────────
    log.info("Loading tokeniser for '%s' …", args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )
    # Decoder models often lack a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("pad_token set to eos_token ('%s')", tokenizer.eos_token)
    tokenizer.padding_side = "right"

    # Pre-tokenise and cache
    def _tokenise(batch):
        return tokenizer(
            batch[TEXT_COLUMN],
            max_length=args.max_length,
            truncation=True,
            # No padding here — DataCollatorWithPadding handles it dynamically
        )

    log.info("Tokenising (max_length=%d) …", args.max_length)
    cols_to_remove = [c for c in raw_ds["train"].column_names if c != LABEL_COLUMN]
    tok_ds = raw_ds.map(
        _tokenise,
        batched=True,
        batch_size=512,
        remove_columns=cols_to_remove,
        desc="Tokenising",
    )
    # Rename label → labels for Trainer
    tok_ds = tok_ds.rename_column(LABEL_COLUMN, "labels")
    tok_ds.set_format("torch")

    log.info("Tokenisation complete. Columns: %s", tok_ds["train"].column_names)

    # ── 3. Model ─────────────────────────────────────────────────────────────
    log.info("Loading model '%s' …", args.model)

    # Try AutoModelForSequenceClassification (works for most decoder models
    # in recent transformers if the config has a supported architecture)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
        log.info("AutoModelForSequenceClassification loaded successfully.")
    except Exception as e:
        log.error("AutoModelForSequenceClassification failed: %s", e)
        log.error("Please check that transformers supports Qwen3.5-4B for SEQ_CLS.")
        raise

    # Pad token id
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model.config, "get_text_config"):
            try:
                model.config.get_text_config().pad_token_id = tokenizer.pad_token_id
            except Exception:
                pass
        if hasattr(model.config, "text_config"):
            try:
                model.config.text_config.pad_token_id = tokenizer.pad_token_id
            except Exception:
                pass

    # Gradient checkpointing (must come before LoRA)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    # ── 4. LoRA ──────────────────────────────────────────────────────────────
    from peft import LoraConfig, TaskType, get_peft_model

    # Detect the actual classification head name
    # For Qwen2/Qwen3 with SEQ_CLS it is typically 'score'
    head_name = None
    for name, _ in model.named_modules():
        if name in ("score", "classifier", "cls_head", "linear"):
            head_name = name
            break
    if head_name is None:
        # fallback — check last Linear layer name
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                head_name = name
        log.warning("Could not detect head name directly; using last Linear: '%s'", head_name)
    else:
        log.info("Classification head detected: '%s'", head_name)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=[head_name] if head_name else ["score"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info("Trainable params: %s / %s (%.2f%%)",
             f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    # ── 5. TrainingArguments ─────────────────────────────────────────────────
    # Checkpoint dir inside saved_models to avoid polluting results/
    ckpt_dir = str(model_save_dir / "checkpoints")

    eval_steps  = 50 if args.debug else 300
    save_steps  = eval_steps
    log_steps   = 10 if args.debug else 25
    max_steps   = 50 if args.debug else -1  # hard cap in debug mode

    # Calculate warmup_steps equivalent to warmup_ratio=0.05 (transformers 5.x uses warmup_steps)
    effective_batch = args.batch_size * args.grad_accum
    total_steps = 50 if args.debug else math.ceil(len(tok_ds["train"]) / effective_batch) * args.epochs
    warmup_steps = max(2, int(total_steps * 0.05))

    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        # Batch / gradient
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=args.grad_accum,
        # Mixed precision
        bf16=True,
        bf16_full_eval=False,
        # Epochs / steps
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        # Optimiser & schedule
        optim="adamw_torch_fused",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        # Eval & checkpoint
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        # Logging
        logging_steps=log_steps,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        # Windows / reproducibility
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
        push_to_hub=False,
        # Disable torch.compile
        torch_compile=False,
    )

    # ── 6. Data collator ────────────────────────────────────────────────────
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    # ── 7. Trainer ───────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    live_cb = LiveLogCallback(log_dir=log_dir, start_time=start_time)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["validation"],
        compute_metrics=compute_metrics,
        data_collator=collator,
        callbacks=[live_cb],
    )

    # ── 8. Debug: estimate time ──────────────────────────────────────────────
    if args.debug:
        log.info("[DEBUG] Running 50-step debug run …")

    log.info("Starting training …")
    train_result = trainer.train()
    train_time_sec = time.time() - start_time
    train_time_min = train_time_sec / 60

    log.info("Training complete in %.1f min (%.0f sec)", train_time_min, train_time_sec)

    peak_vram = peak_vram_gb()
    log.info("Peak VRAM: %.2f GB", peak_vram)

    # Speed estimate
    if args.debug:
        steps_done = trainer.state.global_step
        if steps_done > 0:
            sec_per_step = train_time_sec / steps_done
            total_train_rows = len(raw_ds["train"]) if not args.debug else 498
            effective_batch  = args.batch_size * args.grad_accum
            total_steps_epoch = math.ceil(total_train_rows / effective_batch)
            # Use full dataset size for estimate
            full_train_rows = 48114
            total_steps_full = math.ceil(full_train_rows / effective_batch)
            est_min = (sec_per_step * total_steps_full * args.epochs) / 60
            log.info("=" * 60)
            log.info("SPEED ESTIMATE")
            log.info("  sec/step   : %.2f", sec_per_step)
            log.info("  full steps : %d (batch=%d)", total_steps_full, effective_batch)
            log.info("  estimated  : %.1f min for %d epoch(s) on full dataset",
                     est_min, args.epochs)
            if est_min > 90:
                log.warning("ESTIMATE > 90 min — consider lowering --max_length to 96.")
            elif est_min < 45:
                log.info("ESTIMATE < 45 min — consider using --epochs 2.")
            log.info("=" * 60)

    # ── 9. Best model ────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    best_ckpt   = ""
    for entry in trainer.state.log_history:
        if "eval_f1_macro" in entry:
            if entry["eval_f1_macro"] > best_val_f1:
                best_val_f1 = entry["eval_f1_macro"]
                best_ckpt   = entry.get("checkpoint_path", ckpt_dir)

    log.info("Best val macro F1: %.4f", best_val_f1)

    # ── 10. Save adapter + tokeniser ─────────────────────────────────────────
    log.info("Saving adapter + tokeniser to '%s' …", model_save_dir)
    trainer.save_model(str(model_save_dir))  # saves adapter_config + weights
    tokenizer.save_pretrained(str(model_save_dir))

    # ── 11. Save train_meta.json ──────────────────────────────────────────────
    meta = {
        "model":        args.model,
        "output_name":  output_name,
        "max_length":   args.max_length,
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "grad_accum":   args.grad_accum,
        "lr":           args.lr,
        "lora_r":       args.lora_r,
        "lora_alpha":   args.lora_alpha,
        "seed":         args.seed,
        "train_time_min":   round(train_time_min, 2),
        "best_val_f1_macro":round(best_val_f1, 6),
        "gpu_name":     gpu_name,
        "peak_vram_gb": round(peak_vram, 3),
        "model_type":   "LoRA_SEQ_CLS",   # signal for domain_eval / code_mixed_eval
        "head_name":    head_name or "score",
        "label_names":  LABEL_NAMES,
        "id2label":     ID2LABEL,
    }
    with open(model_save_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("train_meta.json saved.")

    # ── 12. Copy HF trainer_state.json ───────────────────────────────────────
    # The Trainer writes this inside the checkpoint dir; copy to log dir.
    for candidate in [
        Path(ckpt_dir) / "trainer_state.json",
        Path(ckpt_dir) / f"checkpoint-{trainer.state.global_step}" / "trainer_state.json",
    ]:
        if candidate.exists():
            shutil.copy2(candidate, log_dir / "trainer_state.json")
            log.info("trainer_state.json copied to log dir.")
            break
    else:
        # Write it directly from trainer.state
        with open(log_dir / "trainer_state.json", "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, indent=2)

    # ── 13. Plots ─────────────────────────────────────────────────────────────
    plot_loss_curves(log_dir, trainer.state.log_history, output_name)
    plot_val_f1_accuracy(log_dir, trainer.state.log_history)
    plot_lr_schedule(log_dir, trainer.state.log_history)
    log.info("Plots saved.")

    # ── 14. best_checkpoint.txt ───────────────────────────────────────────────
    # Find best checkpoint directory
    best_dir = trainer.state.best_model_checkpoint or ckpt_dir
    with open(log_dir / "best_checkpoint.txt", "w", encoding="utf-8") as f:
        f.write(f"best_step: {trainer.state.best_metric or 0}\n")
        f.write(f"best_val_f1: {best_val_f1:.6f}\n")
        f.write(f"best_checkpoint_path: {best_dir}\n")

    # ── 15. all_runs.csv ──────────────────────────────────────────────────────
    all_runs_path = RESULTS_DIR / "all_runs.csv"
    run_row = {
        "run_name":        output_name,
        "model":           args.model,
        "max_length":      args.max_length,
        "epochs":          args.epochs,
        "lr":              args.lr,
        "batch":           args.batch_size,
        "train_time_min":  round(train_time_min, 2),
        "peak_vram_gb":    round(peak_vram, 3),
        "best_val_f1":     round(best_val_f1, 6),
        "test_f1":         "",   # filled by evaluate_cls_lora.py
        "test_accuracy":   "",
        "latency_ms":      "",
    }
    write_header = not all_runs_path.exists()
    with open(all_runs_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(run_row)
    log.info("Row appended to %s", all_runs_path)

    log.info("=" * 60)
    log.info("TRAINING COMPLETE")
    log.info("  Train time   : %.1f min", train_time_min)
    log.info("  Peak VRAM    : %.2f GB", peak_vram)
    log.info("  Best val F1  : %.4f  (%.2f%%)", best_val_f1, best_val_f1 * 100)
    log.info("  Saved to     : %s", model_save_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()

