"""
evaluate_cls_lora.py
====================
Post-training evaluation for the Qwen3.5-4B LoRA + classification head
saved by train_cls_lora.py.

Loads the base model in bf16, attaches the LoRA adapter, then merges it
before running inference on the **test** split.

Generates
---------
results/qwen3.5-4b-cls-lora_eval_summary.csv
results/qwen3.5-4b-cls-lora_per_class_metrics.csv
results/qwen3.5-4b-cls-lora_confusion_matrix.png
results/qwen3.5-4b-cls-lora/logs/eval_console.log
results/qwen3.5-4b-cls-lora/logs/test_predictions.csv
results/qwen3.5-4b-cls-lora/logs/misclassified.csv
results/qwen3.5-4b-cls-lora/logs/classification_report.txt

Usage
-----
python src/evaluate_cls_lora.py --model_dir saved_models/qwen3.5-4b-cls-lora \\
    --data_dir data/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import LABEL_NAMES, TEXT_COLUMN, LABEL_COLUMN, load_raw_dataset
from train import detect_device

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32  # same as evaluate.py


def setup_eval_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  [%(levelname)s]  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(ch)
    fh = logging.FileHandler(log_dir / "eval_console.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(fh)
    return logging.getLogger(__name__)


def load_model_and_tokenizer(model_dir: str, device: str, log: logging.Logger):
    """
    Load the base model in bf16, attach the LoRA adapter, merge and unload,
    then move to device.

    Returns (model, tokenizer, meta_dict)
    """
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = Path(model_dir).resolve()

    # Read train_meta.json for the base model name
    meta_path = model_path / "train_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    base_model_name = meta.get("model", "Qwen/Qwen3.5-4B")
    log.info("Base model  : %s", base_model_name)
    log.info("Adapter dir : %s", model_path)

    # Print GPU info
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("GPU  : %s  (%.1f GB VRAM)", gpu_name, vram_gb)

    log.info("Loading tokeniser …")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    log.info("Loading base model in bf16 …")
    try:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=3,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
    except Exception as e:
        log.warning("sdpa failed (%s), retrying without attn_implementation …", e)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=3,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )

    if tokenizer.pad_token_id is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(base_model.config, "get_text_config"):
            try:
                base_model.config.get_text_config().pad_token_id = tokenizer.pad_token_id
            except Exception:
                pass
        if hasattr(base_model.config, "text_config"):
            try:
                base_model.config.text_config.pad_token_id = tokenizer.pad_token_id
            except Exception:
                pass

    log.info("Attaching LoRA adapter …")
    peft_model = PeftModel.from_pretrained(base_model, str(model_path))

    log.info("Merging adapter into base model …")
    model = peft_model.merge_and_unload()
    model.eval()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %s (%.1f M)", f"{n_params:,}", n_params / 1e6)
    return model, tokenizer, meta


def run_inference(
    model,
    tokenizer,
    texts: list[str],
    max_length: int,
    device: str,
    log: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Run batched inference. Returns (all_probs, all_preds, ms_per_sample_batch, ms_per_sample_single).
    CUDA warmup included.
    """
    # --- CUDA warmup ---
    if device == "cuda":
        log.info("CUDA warmup (3 passes) …")
        dummy = tokenizer(
            ["warmup"] * min(4, BATCH_SIZE),
            return_tensors="pt", max_length=max_length,
            truncation=True, padding=True,
        )
        dummy = {k: v.to(device) for k, v in dummy.items()}
        with torch.no_grad():
            for _ in range(3):
                model(**dummy)
        torch.cuda.synchronize()

    all_probs: list[np.ndarray] = []
    total_time = 0.0
    n_samples  = 0

    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i: i + BATCH_SIZE]
            enc   = tokenizer(
                batch, max_length=max_length,
                truncation=True, padding=True, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(**enc).logits
            if device == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0
            n_samples  += len(batch)

            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            all_probs.append(probs)

            if (i // BATCH_SIZE) % 10 == 0:
                log.info("  %d / %d done", min(i + BATCH_SIZE, len(texts)), len(texts))

    all_probs_arr = np.concatenate(all_probs, axis=0)
    all_preds = all_probs_arr.argmax(axis=-1)
    ms_per_sample_batch = (total_time / n_samples) * 1000 if n_samples else 0.0

    # --- batch=1 latency ---
    log.info("Measuring batch=1 latency …")
    lat1_times = []
    sample_text = texts[:10]
    with torch.no_grad():
        for txt in sample_text:
            enc = tokenizer(
                [txt], max_length=max_length,
                truncation=True, padding=True, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(**enc)
            if device == "cuda":
                torch.cuda.synchronize()
            lat1_times.append(time.perf_counter() - t0)
    ms_per_sample_single = (sum(lat1_times) / len(lat1_times)) * 1000 if lat1_times else 0.0

    log.info("Latency (batch=%d) : %.2f ms/sample", BATCH_SIZE, ms_per_sample_batch)
    log.info("Latency (batch=1)  : %.2f ms/sample", ms_per_sample_single)
    return all_probs_arr, all_preds, ms_per_sample_batch, ms_per_sample_single


def save_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    all_probs: np.ndarray,
    texts: list[str],
    ms_per_sample_batch: float,
    ms_per_sample_single: float,
    tag: str,
    log_dir: Path,
    log: logging.Logger,
) -> dict:
    """Compute + save all eval outputs. Returns summary dict."""
    macro_f1  = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    macro_prec= precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro",    zero_division=0)
    acc       = accuracy_score(y_true, y_pred)

    log.info("=" * 60)
    log.info("TEST EVALUATION RESULTS")
    log.info("=" * 60)
    log.info("  Macro F1        : %.4f  (%.2f%%)", macro_f1,   macro_f1   * 100)
    log.info("  Macro Precision : %.4f", macro_prec)
    log.info("  Macro Recall    : %.4f", macro_rec)
    log.info("  Accuracy        : %.4f  (%.2f%%)", acc, acc * 100)
    log.info("  Latency batch=%d: %.2f ms/sample", BATCH_SIZE, ms_per_sample_batch)
    log.info("  Latency batch=1 : %.2f ms/sample", ms_per_sample_single)
    log.info("-" * 60)
    report_str = classification_report(
        y_true, y_pred, target_names=LABEL_NAMES, zero_division=0
    )
    log.info("\n%s", report_str)

    # --- classification_report.txt ---
    with open(log_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_str)

    # --- eval summary CSV ---
    summary = {
        "macro_f1":           round(macro_f1, 6),
        "macro_precision":    round(macro_prec, 6),
        "macro_recall":       round(macro_rec, 6),
        "accuracy":           round(acc, 6),
        "latency_ms_sample":  round(ms_per_sample_batch, 3),
        "latency_ms_batch1":  round(ms_per_sample_single, 3),
    }
    pd.DataFrame([summary]).to_csv(
        RESULTS_DIR / f"{tag}_eval_summary.csv", index=False
    )

    # --- per-class metrics CSV ---
    report_dict = classification_report(
        y_true, y_pred, target_names=LABEL_NAMES,
        zero_division=0, output_dict=True,
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        RESULTS_DIR / f"{tag}_per_class_metrics.csv"
    )

    # --- confusion matrix PNG ---
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix — {tag}", fontsize=14, fontweight="bold")
    sns.heatmap(cm,      annot=True, fmt="d",    cmap="Blues",   ax=axes[0],
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, linewidths=0.5)
    axes[0].set_title("Raw Counts"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[1],
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title("Row-Normalised"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    plt.tight_layout()
    cm_path = RESULTS_DIR / f"{tag}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Confusion matrix saved → %s", cm_path)

    # --- test_predictions.csv ---
    pred_df = pd.DataFrame({
        "text":       texts,
        "true_label": [LABEL_NAMES[i] for i in y_true],
        "pred_label": [LABEL_NAMES[i] for i in y_pred],
        "confidence": [round(float(all_probs[j, y_pred[j]]), 5) for j in range(len(y_pred))],
        "prob_negative": [round(float(all_probs[j, 0]), 5) for j in range(len(y_pred))],
        "prob_neutral":  [round(float(all_probs[j, 1]), 5) for j in range(len(y_pred))],
        "prob_positive": [round(float(all_probs[j, 2]), 5) for j in range(len(y_pred))],
    })
    pred_df.to_csv(log_dir / "test_predictions.csv", index=False)

    # --- misclassified.csv ---
    mis_df = pred_df[pred_df["true_label"] != pred_df["pred_label"]].copy()
    mis_df.to_csv(log_dir / "misclassified.csv", index=False)
    log.info("Predictions saved — %d misclassified / %d total (%.1f%% error)",
             len(mis_df), len(pred_df), 100 * len(mis_df) / len(pred_df))

    return summary


def update_all_runs(output_name: str, summary: dict, log: logging.Logger) -> None:
    """Append / update test metrics in results/all_runs.csv."""
    all_runs_path = RESULTS_DIR / "all_runs.csv"
    if not all_runs_path.exists():
        log.warning("all_runs.csv not found — skipping update.")
        return
    df = pd.read_csv(all_runs_path)
    mask = df["run_name"] == output_name
    if mask.any():
        df.loc[mask, "test_f1"]      = summary["macro_f1"]
        df.loc[mask, "test_accuracy"]= summary["accuracy"]
        df.loc[mask, "latency_ms"]   = summary["latency_ms_sample"]
        df.to_csv(all_runs_path, index=False)
        log.info("all_runs.csv updated with test metrics.")
    else:
        log.warning("Run '%s' not found in all_runs.csv.", output_name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Qwen3.5-4B LoRA cls head on MahaSent-MD test set."
    )
    p.add_argument("--model_dir",  type=str, default="saved_models/qwen3.5-4b-cls-lora")
    p.add_argument("--data_dir",   type=str, default="data/")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--debug",      action="store_true")
    p.add_argument("--cpu",        action="store_true")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    device  = detect_device(force_cpu=args.cpu)

    model_path  = Path(args.model_dir)
    output_name = model_path.name          # "qwen3.5-4b-cls-lora"
    tag         = output_name              # used as file prefix
    log_dir     = RESULTS_DIR / output_name / "logs"

    log = setup_eval_logging(log_dir)
    log.info("=" * 60)
    log.info("evaluate_cls_lora.py  —  %s", output_name)
    log.info("=" * 60)

    # Reload max_length from meta if not overridden
    meta_path = model_path / "train_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if args.max_length == 128:  # default → use trained value
            args.max_length = meta.get("max_length", 128)

    # Load model
    model, tokenizer, meta = load_model_and_tokenizer(
        str(model_path), device, log
    )

    # Load test data
    log.info("Loading test set from '%s' …", args.data_dir)
    raw_ds = load_raw_dataset(debug=args.debug, data_dir=args.data_dir)
    test_split = raw_ds["test"]
    texts  = list(test_split[TEXT_COLUMN])
    y_true = np.array(test_split[LABEL_COLUMN], dtype=np.int64)
    log.info("Test set: %d samples", len(texts))

    # Inference
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    all_probs, y_pred, ms_batch, ms_single = run_inference(
        model, tokenizer, texts, args.max_length, device, log
    )
    peak_vram = torch.cuda.max_memory_allocated(0) / 1e9 if device == "cuda" else 0.0
    log.info("Peak VRAM during eval: %.2f GB", peak_vram)

    # Save results
    summary = save_results(
        y_true, y_pred, all_probs, texts,
        ms_batch, ms_single, tag, log_dir, log
    )

    # Update all_runs.csv
    update_all_runs(output_name, summary, log)

    log.info("=" * 60)
    log.info("EVALUATION COMPLETE")
    log.info("  Test Macro F1  : %.4f  (%.2f%%)",
             summary["macro_f1"], summary["macro_f1"] * 100)
    log.info("  Test Accuracy  : %.4f  (%.2f%%)",
             summary["accuracy"], summary["accuracy"] * 100)
    log.info("  Latency (b=%d) : %.2f ms/sample", BATCH_SIZE, summary["latency_ms_sample"])
    log.info("  Latency (b=1)  : %.2f ms/sample", summary["latency_ms_batch1"])
    log.info("  MuRIL 81.20%% %s",
             "BEATEN! ✓" if summary["macro_f1"] > 0.8120 else "not beaten yet.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

