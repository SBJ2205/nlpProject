"""
evaluate_sarvam.py
==================
Evaluation and benchmarking script for `sarvamai/sarvam-1` (2B) on MahaSent test set.

Key Features:
- Evaluates test set (MahaSent_All_Test.csv) exclusively for benchmarking
- Supports evaluating both Base Sarvam-1 (zero-shot) and Fine-tuned Sarvam-1 (LoRA)
- Prints a side-by-side performance comparison table (Original vs Fine-tuned)
- Computes Accuracy, Precision, Recall, Macro-F1, and per-sample latency
- Saves outputs to results/:
  - sarvam-1_eval_summary.csv
  - sarvam-1_comparison.csv
  - sarvam-1_per_class_metrics.csv
  - sarvam-1_confusion_matrix.png
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent))
from auto_config import get_auto_config
from data_loader import LABEL_COLUMN, LABEL_NAMES, TEXT_COLUMN, load_raw_dataset
from gpu_detector import print_hardware_report

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_BASE_MODEL: str = "sarvamai/sarvam-1"
DEFAULT_ADAPTER_DIR: str = "saved_models/sarvam-1-lora"
RESULTS_DIR: Path = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CACHE_DIR: str = str(Path(__file__).parent.parent / "model")
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR

PROMPT_TEMPLATE: str = (
    "Classify the sentiment of the following Marathi text. Reply with ONLY ONE WORD from [Positive, Negative, Neutral].\n"
    "Text: {text_sample}\n"
    "Sentiment:"
)

CLASS_INDEX_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def parse_generated_sentiment(text: str) -> tuple[int, str]:
    """Parse raw generation into label index (0, 1, 2) and class name."""
    cleaned = text.strip().lower()
    tokens = re.findall(r"\b\w+\b", cleaned)
    first_token = tokens[0] if tokens else cleaned

    if "pos" in first_token or "आवड" in cleaned or "chan" in cleaned:
        return 2, "positive"
    elif "neg" in first_token or "वाईट" in cleaned or "bakwaas" in cleaned:
        return 0, "negative"
    elif "neu" in first_token or "मध्य" in cleaned or "thik" in cleaned:
        return 1, "neutral"

    if "positive" in cleaned:
        return 2, "positive"
    if "negative" in cleaned:
        return 0, "negative"
    if "neutral" in cleaned:
        return 1, "neutral"

    return 1, "neutral"


def load_model_and_tokenizer(
    base_model_name: str = DEFAULT_BASE_MODEL,
    adapter_dir: Optional[str] = None,
    force_cpu: bool = False,
):
    """Load base model with optional attached LoRA adapter."""
    use_cuda = torch.cuda.is_available() and not force_cpu
    log.info("Loading tokenizer for '%s' …", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, cache_dir=MODEL_CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    log.info("Loading model '%s' (CUDA=%s) …", base_model_name, use_cuda)
    if use_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR,
        )

    if adapter_dir:
        adapter_path = Path(adapter_dir)
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            if PeftModel is None:
                log.warning("peft not installed — unable to load LoRA adapter.")
            else:
                log.info("Attaching LoRA adapter from '%s' …", adapter_path)
                model = PeftModel.from_pretrained(model, str(adapter_path))
        else:
            log.warning("Adapter directory '%s' not found. Evaluating in base mode.", adapter_dir)

    model.eval()
    return model, tokenizer


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    output_path: Path,
    title: str = "Sarvam-1 (QLoRA)",
) -> None:
    """Save raw-count and row-normalized confusion matrix heatmaps."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix — {title}", fontsize=14, fontweight="bold")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=axes[0],
        linewidths=0.5,
    )
    axes[0].set_title("Raw Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=axes[1],
        linewidths=0.5,
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title("Row-Normalised (Recall per Class)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Confusion matrix plot saved → %s", output_path)


def run_evaluation_for_model(
    model,
    tokenizer,
    texts: list[str],
    y_true: np.ndarray,
    batch_size: int = 4,
    device: str = "cuda",
) -> Tuple[Dict[str, float], np.ndarray]:
    """Run generation and calculate evaluation metrics for a loaded model."""
    all_preds = []
    total_time = 0.0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        prompts = [PROMPT_TEMPLATE.format(text_sample=t) for t in batch_texts]

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        input_lens = attention_mask.sum(dim=1)

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        total_time += time.perf_counter() - t0

        for gen, in_len in zip(generated_ids, input_lens):
            new_tokens = gen[in_len:]
            resp_str = tokenizer.decode(new_tokens, skip_special_tokens=True)
            pred_idx, _ = parse_generated_sentiment(resp_str)
            all_preds.append(pred_idx)

    y_pred = np.array(all_preds, dtype=np.int64)
    ms_per_sample = (total_time / len(texts)) * 1000 if texts else 0.0

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    metrics = {
        "Accuracy": round(float(acc), 4),
        "Macro-F1": round(float(macro_f1), 4),
        "Precision": round(float(macro_prec), 4),
        "Recall": round(float(macro_rec), 4),
        "Latency_ms": round(float(ms_per_sample), 2),
    }

    return metrics, y_pred


def evaluate_sarvam(
    base_model_name: str = DEFAULT_BASE_MODEL,
    adapter_dir: str = DEFAULT_ADAPTER_DIR,
    data_dir: Optional[str] = None,
    debug: bool = False,
    force_cpu: bool = False,
    batch_size: int = 4,
    compare_baseline: bool = True,
) -> dict:
    """Run evaluation on MahaSent_All_Test.csv and compare baseline vs fine-tuned."""
    profile = print_hardware_report()
    device = "cuda" if (profile.gpu_available and not force_cpu) else "cpu"

    log.info("Loading MahaSent TEST dataset from '%s' …", data_dir or "default location")
    raw_ds = load_raw_dataset(debug=debug, data_dir=data_dir)
    test_ds = raw_ds["test"]

    texts = test_ds[TEXT_COLUMN]
    y_true = np.array(test_ds[LABEL_COLUMN], dtype=np.int64)

    results = {}

    # 1. Evaluate Fine-Tuned Model
    log.info("--- Evaluating Fine-Tuned Sarvam-1 (LoRA) ---")
    model_ft, tokenizer_ft = load_model_and_tokenizer(
        base_model_name=base_model_name,
        adapter_dir=adapter_dir,
        force_cpu=force_cpu,
    )
    ft_metrics, ft_preds = run_evaluation_for_model(
        model_ft, tokenizer_ft, texts, y_true, batch_size=batch_size, device=device
    )
    results["Fine-tuned"] = ft_metrics

    # Cleanup fine-tuned model from memory
    del model_ft, tokenizer_ft
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Evaluate Baseline Model (Zero-shot base model)
    if compare_baseline:
        log.info("--- Evaluating Base Sarvam-1 (Zero-shot Original) ---")
        model_base, tokenizer_base = load_model_and_tokenizer(
            base_model_name=base_model_name,
            adapter_dir=None,
            force_cpu=force_cpu,
        )
        base_metrics, base_preds = run_evaluation_for_model(
            model_base, tokenizer_base, texts, y_true, batch_size=batch_size, device=device
        )
        results["Original"] = base_metrics

        del model_base, tokenizer_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. Print Side-by-Side Comparison Table
    print("\n==================================================")
    print(" BASELINE VS FINE-TUNED MODEL COMPARISON")
    print("==================================================")
    if compare_baseline:
        print(f"{'Metric':<18} {'Original':<15} {'Fine-tuned':<15}")
        print("-" * 50)
        for metric_name in ["Accuracy", "Macro-F1", "Precision", "Recall"]:
            orig_val = results["Original"][metric_name]
            ft_val = results["Fine-tuned"][metric_name]
            print(f"{metric_name:<18} {orig_val:<15} {ft_val:<15}")
    else:
        print(f"{'Metric':<18} {'Fine-tuned':<15}")
        print("-" * 35)
        for metric_name in ["Accuracy", "Macro-F1", "Precision", "Recall"]:
            ft_val = results["Fine-tuned"][metric_name]
            print(f"{metric_name:<18} {ft_val:<15}")
    print("==================================================\n")

    # 4. Save Outputs & Plots
    comp_df = pd.DataFrame(results)
    comp_path = RESULTS_DIR / "sarvam-1_comparison.csv"
    comp_df.to_csv(comp_path)
    log.info("Saved baseline comparison → %s", comp_path)

    cm_path = RESULTS_DIR / "sarvam-1_confusion_matrix.png"
    plot_confusion_matrix(y_true, ft_preds, LABEL_NAMES, cm_path, title="Sarvam-1 (QLoRA Fine-tuned)")

    report_dict = classification_report(
        y_true, ft_preds, target_names=LABEL_NAMES, zero_division=0, output_dict=True
    )
    per_class_path = RESULTS_DIR / "sarvam-1_per_class_metrics.csv"
    pd.DataFrame(report_dict).transpose().to_csv(per_class_path)
    log.info("Saved per-class metrics → %s", per_class_path)

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Sarvam-1 on MahaSent test set.")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help=f"Base model name (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--adapter_dir", type=str, default=DEFAULT_ADAPTER_DIR, help=f"Adapter path (default: {DEFAULT_ADAPTER_DIR})")
    parser.add_argument("--data_dir", type=str, default=None, help="Local CSV data directory")
    parser.add_argument("--debug", action="store_true", help="Evaluate on 99-sample debug subset")
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    parser.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip evaluating original base model")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_sarvam(
        base_model_name=args.base_model,
        adapter_dir=args.adapter_dir,
        data_dir=args.data_dir,
        debug=args.debug,
        force_cpu=args.cpu,
        batch_size=args.batch_size,
        compare_baseline=not args.skip_baseline,
    )
