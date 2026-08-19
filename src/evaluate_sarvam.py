"""
evaluate_sarvam.py
==================
Evaluation and benchmarking script for the fine-tuned `sarvamai/sarvam-1` (2B)
Generative Small Language Model (SLM) on the MahaSent-MD test set.

Key Features:
- Loads base Sarvam-1 model and attaches the fine-tuned LoRA adapter from `saved_models/sarvam-1-lora/`
- Performs causal generation with prompt prefix slicing to isolate generated prediction tokens
- Robust response parsing into categorical classes: [Negative, Neutral, Positive]
- Computes Macro F1, Precision, Recall, Accuracy, and per-sample latency
- Saves benchmark outputs to `results/`:
  - `results/sarvam-1_eval_summary.csv`
  - `results/sarvam-1_per_class_metrics.csv`
  - `results/sarvam-1_confusion_matrix.png`

Usage:
------
# Full evaluation on GPU
python src/evaluate_sarvam.py --data_dir data/

# Fast debug test (99 samples)
python src/evaluate_sarvam.py --data_dir data/ --debug
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

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

# Make src/ importable when called from project root
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import LABEL_COLUMN, LABEL_NAMES, TEXT_COLUMN, load_raw_dataset

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BASE_MODEL: str = "sarvamai/sarvam-1"
DEFAULT_ADAPTER_DIR: str = "saved_models/sarvam-1-lora"
RESULTS_DIR: Path = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_TEMPLATE: str = (
    "Classify the sentiment of the following Marathi text. Reply with ONLY ONE WORD from [Positive, Negative, Neutral].\n"
    "Text: {text_sample}\n"
    "Sentiment:"
)

# Label index to Name mapping
# 0 -> negative, 1 -> neutral, 2 -> positive
CLASS_INDEX_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------

def parse_generated_sentiment(text: str) -> tuple[int, str]:
    """
    Parse the raw generated output into a class index (0, 1, 2) and name.
    
    Returns:
        (class_idx, class_name)
    """
    cleaned = text.strip().lower()
    # Match words or prefixes
    tokens = re.findall(r"\b\w+\b", cleaned)
    first_token = tokens[0] if tokens else cleaned

    if "pos" in first_token or "आवड" in cleaned or "chan" in cleaned:
        return 2, "positive"
    elif "neg" in first_token or "वाईट" in cleaned or "रूट" in cleaned or "bakwaas" in cleaned:
        return 0, "negative"
    elif "neu" in first_token or "मध्य" in cleaned or "thik" in cleaned:
        return 1, "neutral"

    # Fallback to substring scanning across full cleaned output
    if "positive" in cleaned:
        return 2, "positive"
    if "negative" in cleaned:
        return 0, "negative"
    if "neutral" in cleaned:
        return 1, "neutral"

    # Default neutral fallback if model outputs ambiguous token
    return 1, "neutral"


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------

def load_sarvam_model(
    base_model_name: str = DEFAULT_BASE_MODEL,
    adapter_dir: str = DEFAULT_ADAPTER_DIR,
    force_cpu: bool = False,
):
    """Load base model and attach LoRA adapter if present."""
    use_cuda = torch.cuda.is_available() and not force_cpu
    log.info("Loading tokenizer for '%s' …", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    log.info("Loading base model '%s' (CUDA=%s) …", base_model_name, use_cuda)
    if use_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    adapter_path = Path(adapter_dir)
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        if PeftModel is None:
            log.warning("peft library not installed — unable to load LoRA adapter.")
        else:
            log.info("Attaching LoRA adapter from '%s' …", adapter_path)
            model = PeftModel.from_pretrained(model, str(adapter_path))
    else:
        log.warning("Adapter directory '%s' not found. Evaluating base model in zero-shot mode.", adapter_dir)

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Confusion Matrix Plot
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------------------

def evaluate_sarvam(
    base_model_name: str = DEFAULT_BASE_MODEL,
    adapter_dir: str = DEFAULT_ADAPTER_DIR,
    data_dir: str = None,
    debug: bool = False,
    force_cpu: bool = False,
    batch_size: int = 4,
) -> dict:
    """Run full evaluation and save reports to results/."""
    # 1. Load test data
    raw_ds = load_raw_dataset(debug=debug, data_dir=data_dir)
    test_key = "test" if "test" in raw_ds else "validation"
    test_ds = raw_ds[test_key]

    texts = test_ds[TEXT_COLUMN]
    raw_labels = test_ds[LABEL_COLUMN]
    y_true = np.array(raw_labels, dtype=np.int64)

    log.info("Evaluating on %d samples from '%s' split …", len(texts), test_key)

    # 2. Load model
    model, tokenizer = load_sarvamModel(
        base_model_name=base_model_name,
        adapter_dir=adapter_dir,
        force_cpu=force_cpu,
    )

    device = "cuda" if (torch.cuda.is_available() and not force_cpu) else "cpu"

    # 3. Batch generation & output slicing
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

        if (i // batch_size) % 10 == 0:
            log.info("  %d / %d samples processed", min(i + batch_size, len(texts)), len(texts))

    y_pred = np.array(all_preds, dtype=np.int64)
    ms_per_sample = (total_time / len(texts)) * 1000 if texts else 0.0

    # 4. Compute metrics
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    log.info("=" * 60)
    log.info("SARVAM-1 EVALUATION RESULTS")
    log.info("=" * 60)
    log.info("  Accuracy        : %.4f (%.2f%%)", acc, acc * 100)
    log.info("  Macro F1        : %.4f", macro_f1)
    log.info("  Macro Precision : %.4f", macro_prec)
    log.info("  Macro Recall    : %.4f", macro_rec)
    log.info("  Inference Latency: %.2f ms / sample", ms_per_sample)
    log.info("-" * 60)
    log.info(
        "\n%s",
        classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0),
    )

    # 5. Save outputs
    summary = {
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "accuracy": round(float(acc), 4),
        "ms_per_sample": round(float(ms_per_sample), 3),
    }
    summary_path = RESULTS_DIR / "sarvam-1_eval_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    log.info("Summary saved → %s", summary_path)

    report_dict = classification_report(
        y_true, y_pred, target_names=LABEL_NAMES, zero_division=0, output_dict=True
    )
    per_class_path = RESULTS_DIR / "sarvam-1_per_class_metrics.csv"
    pd.DataFrame(report_dict).transpose().to_csv(per_class_path)
    log.info("Per-class metrics saved → %s", per_class_path)

    cm_path = RESULTS_DIR / "sarvam-1_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, LABEL_NAMES, cm_path, title="sarvamai--sarvam-1-lora")

    return summary


# ---------------------------------------------------------------------------
# Helper function name alias for backwards compatibility
# ---------------------------------------------------------------------------
load_sarvamModel = load_sarvam_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Sarvam-1 on Marathi sentiment.")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help=f"Base model name (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--adapter_dir", type=str, default=DEFAULT_ADAPTER_DIR, help=f"LoRA adapter path (default: {DEFAULT_ADAPTER_DIR})")
    parser.add_argument("--data_dir", type=str, default=None, help="Local CSV data directory")
    parser.add_argument("--debug", action="store_true", help="Evaluate on 99-sample debug subset")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size")
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
    )
