"""
train.py
========
Fine-tune a multilingual BERT (indic-bert or MuRIL) on MahaSent-MD.

Usage
-----
# Debug run — 99-row stratified sample, 1 epoch, CPU (fast sanity-check)
python src/train.py --data_dir data/ --debug --cpu

# Full fine-tune with indic-bert on CPU
python src/train.py --data_dir data/ --model ai4bharat/indic-bert --cpu

# Full fine-tune with GPU (after fixing PyTorch sm_120 support)
python src/train.py --data_dir data/ --model ai4bharat/indic-bert

# Use MuRIL instead (public, no HuggingFace login needed)
python src/train.py --data_dir data/ --model google/muril-base-cased
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Make src/ importable when called from project root
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import LABEL_COLUMN, get_tokenised_datasets, load_raw_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = "google/muril-base-cased"   # public, no login needed
RESULTS_DIR: str = "results"
SEED: int = 42


# ---------------------------------------------------------------------------
# Loss curve plotting  (Objectives 10 & 11)
# ---------------------------------------------------------------------------

def plot_loss_curves(log_history: list[dict], output_dir: str, model_slug: str) -> None:
    """
    Save a training / validation loss curve PNG from the Trainer log history.

    Parameters
    ----------
    log_history : list[dict]
        ``trainer.state.log_history`` after training.
    output_dir : str
        Base output directory — plot saved to ``results/``.
    model_slug : str
        Used as part of the filename, e.g. ``ai4bharat--indic-bert``.
    """
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
        log.warning("No loss history found — skipping loss curve plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_steps, train_losses, label="Training Loss",   color="#2196F3", linewidth=2)
    if eval_losses:
        ax.plot(eval_steps, eval_losses, label="Validation Loss", color="#FF5722",
                linewidth=2, linestyle="--", marker="o", markersize=5)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss",          fontsize=12)
    ax.set_title(f"Training & Validation Loss — {model_slug}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out = Path(RESULTS_DIR) / f"{model_slug}_loss_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Loss curves saved → %s", out)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device(force_cpu: bool = False) -> str:
    """
    Detect the best available compute device.

    Parameters
    ----------
    force_cpu : bool
        Always return ``"cpu"`` — useful when the GPU driver (e.g. sm_120
        Blackwell) is not yet supported by the installed PyTorch build.

    Returns
    -------
    str  ``"cuda"`` or ``"cpu"``
    """
    if force_cpu:
        log.info("CPU mode forced via --cpu flag.")
        return "cpu"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("GPU detected: %s  (%.1f GB VRAM)", name, vram)
        return "cuda"
    log.warning("No CUDA GPU found — training on CPU (will be slow).")
    return "cpu"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics(label_names: list[str]):
    """Return a compute_metrics function compatible with HuggingFace Trainer."""
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    def compute_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "macro_f1": f1_score(
                labels, predictions, average="macro", zero_division=0
            ),
            "macro_precision": precision_score(
                labels, predictions, average="macro", zero_division=0
            ),
            "macro_recall": recall_score(
                labels, predictions, average="macro", zero_division=0
            ),
            "accuracy": accuracy_score(labels, predictions),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------

def build_training_args(
    output_dir: str,
    device: str,
    debug: bool,
) -> TrainingArguments:
    """
    Return TrainingArguments compatible with transformers 5.x.

    Removed params vs 4.x:
      - ``logging_dir``  → logs now go to output_dir automatically
      - ``warmup_ratio`` → use ``warmup_steps`` instead
    ``fp16`` is disabled on CPU automatically.
    """
    return TrainingArguments(
        output_dir=output_dir,
        # ── Batch & gradient ──────────────────────────────────────────────────
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,        # effective batch = 32
        # ── Mixed precision ───────────────────────────────────────────────────
        fp16=(device == "cuda"),              # disabled on CPU
        use_cpu=(device == "cpu"),            # force HuggingFace to respect --cpu
        # ── Epochs ───────────────────────────────────────────────────────────
        num_train_epochs=1 if debug else 5,
        # ── Optimiser & schedule ──────────────────────────────────────────────
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=10,                      # warmup_ratio removed in 5.x
        lr_scheduler_type="cosine",
        # ── Evaluation & checkpointing ────────────────────────────────────────
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        # ── Logging ───────────────────────────────────────────────────────────
        logging_steps=10,                     # logging_dir removed in 5.x
        report_to="none",                     # disable W&B / MLflow
        # ── Reproducibility ───────────────────────────────────────────────────
        seed=SEED,
        data_seed=SEED,
        # ── Windows safety ────────────────────────────────────────────────────
        dataloader_num_workers=0,
        push_to_hub=False,
    )


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(
    model_name: str = DEFAULT_MODEL,
    debug: bool = False,
    force_cpu: bool = False,
    data_dir: str = None,
) -> None:
    """
    Full fine-tuning pipeline: load data → init model → train → save.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier to fine-tune.
    debug : bool
        Use 99-row stratified sample and 1 epoch for a quick sanity-check.
    force_cpu : bool
        Skip GPU — useful when the GPU driver is not yet compatible.
    data_dir : str | None
        Path to local CSV directory; falls back to HuggingFace Hub if None.
    """
    device = detect_device(force_cpu=force_cpu)

    # 1. Load & tokenise
    log.info("Loading tokenised dataset …")
    tok_ds = get_tokenised_datasets(
        model_name=model_name, debug=debug, data_dir=data_dir
    )
    train_split = tok_ds["train"]
    eval_key    = "validation" if "validation" in tok_ds else "test"
    eval_split  = tok_ds[eval_key]

    # 2. Label names (from raw dataset features)
    raw_ds    = load_raw_dataset(debug=debug, data_dir=data_dir)
    raw_train = raw_ds["train"]
    if hasattr(raw_train.features[LABEL_COLUMN], "names"):
        label_names: list[str] = raw_train.features[LABEL_COLUMN].names
    else:
        label_names = [str(c) for c in sorted(set(raw_train[LABEL_COLUMN]))]
    num_labels = len(label_names)
    log.info("Labels: %d  →  %s", num_labels, label_names)

    # 3. Model
    log.info("Loading model '%s' …", model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # Log model size (Objective 11)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model size  : %s parameters (%.1f M total, %.1f M trainable)",
             f"{n_params:,}", n_params / 1e6, n_trainable / 1e6)

    # 4. Output path — derived from model name so different models never collide
    model_slug = model_name.replace("/", "--")
    output_dir = (
        f"outputs/{model_slug}-debug" if debug else f"outputs/{model_slug}"
    )
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    training_args = build_training_args(
        output_dir=output_dir, device=device, debug=debug
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=eval_split,
        compute_metrics=make_compute_metrics(label_names),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 5. Train
    mode_tag = "[DEBUG]" if debug else "[FULL]"
    log.info("%s Starting training — model: %s", mode_tag, model_name)
    train_result = trainer.train()

    # 6. Save model + tokeniser
    log.info("Saving model to '%s' …", output_dir)
    trainer.save_model(output_dir)
    AutoTokenizer.from_pretrained(model_name).save_pretrained(output_dir)

    # 7. Final eval
    log.info("Final evaluation on '%s' split …", eval_key)
    metrics = trainer.evaluate()
    log.info("Final metrics: %s", metrics)

    # 8. Save summary CSV
    import pandas as pd
    summary = {**train_result.metrics, **metrics}
    pd.DataFrame([summary]).to_csv(
        Path(RESULTS_DIR) / "train_metrics.csv", index=False
    )
    log.info("Training summary → %s/train_metrics.csv", RESULTS_DIR)

    # 9. Plot loss curves (Objective 10)
    model_slug = model_name.replace("/", "--")
    plot_loss_curves(trainer.state.log_history, output_dir, model_slug)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune indic-bert / MuRIL on Marathi Sentiment."
    )
    parser.add_argument("--debug", action="store_true",
                        help="1 epoch on 99-row stratified sample (fast sanity-check).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model to fine-tune (default: {DEFAULT_MODEL}).")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU — use when GPU driver is incompatible.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to local CSV directory (skips HuggingFace Hub).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        model_name=args.model,
        debug=args.debug,
        force_cpu=args.cpu,
        data_dir=args.data_dir,
    )
