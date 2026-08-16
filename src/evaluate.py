"""
evaluate.py
===========
Post-training evaluation for the fine-tuned model saved by train.py.

Generates
---------
- Macro F1 / Precision / Recall (console + CSV)
- Per-class classification report (CSV)
- Confusion matrix PNG (raw counts + row-normalised)

Usage
-----
# Evaluate a checkpoint
python src/evaluate.py --data_dir data/ --model_dir outputs/ai4bharat--indic-bert-debug --debug --cpu

# Full evaluation (GPU)
python src/evaluate.py --data_dir data/ --model_dir outputs/ai4bharat--indic-bert
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Make src/ importable when called from project root
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import LABEL_COLUMN, TEXT_COLUMN, load_raw_dataset
from train import detect_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_MODEL_DIR: str = "outputs/ai4bharat--indic-bert"
RESULTS_DIR: Path = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def get_predictions(
    model_dir: str,
    debug: bool = False,
    device: str = "cpu",
    data_dir: str = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load the fine-tuned model and run inference on the validation split.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        ``(y_true, y_pred, label_names)``
    """
    log.info("Loading model from '%s' …", model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Log model size (Objective 11)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model size: %s parameters (%.1f M)", f"{n_params:,}", n_params / 1e6)

    raw_ds   = load_raw_dataset(debug=debug, data_dir=data_dir)
    eval_key = "validation" if "validation" in raw_ds else "test"
    eval_data = raw_ds[eval_key]

    if hasattr(eval_data.features[LABEL_COLUMN], "names"):
        label_names: list[str] = eval_data.features[LABEL_COLUMN].names
    else:
        label_names = [str(c) for c in sorted(set(eval_data[LABEL_COLUMN]))]

    texts:  list[str]  = eval_data[TEXT_COLUMN]
    y_true: np.ndarray = np.array(eval_data[LABEL_COLUMN], dtype=np.int64)

    log.info("Running inference on %d samples …", len(texts))
    BATCH = 64
    all_preds: list[int] = []
    total_time: float = 0.0

    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        enc = tokenizer(
            batch, max_length=128, padding=True,
            truncation=True, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            t0 = time.perf_counter()
            logits = model(**enc).logits
            total_time += time.perf_counter() - t0
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())

        if (i // BATCH) % 5 == 0:
            log.info("  %d / %d done", min(i + BATCH, len(texts)), len(texts))

    ms_per_sample = (total_time / len(texts)) * 1000 if texts else 0.0
    log.info("Inference latency: %.2f ms / sample", ms_per_sample)
    return y_true, np.array(all_preds, dtype=np.int64), label_names, ms_per_sample


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def compute_and_log_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    model_dir: str,
) -> dict:
    """Compute metrics, print them, and save CSVs."""
    macro_f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)

    log.info("=" * 55)
    log.info("EVALUATION RESULTS")
    log.info("=" * 55)
    log.info("  Model           : %s", model_dir)
    log.info("  Macro F1        : %.4f", macro_f1)
    log.info("  Macro Precision : %.4f", macro_prec)
    log.info("  Macro Recall    : %.4f", macro_rec)
    log.info("-" * 55)
    log.info(
        "\n%s",
        classification_report(y_true, y_pred, target_names=label_names,
                               zero_division=0),
    )

    tag = Path(model_dir).name

    # Per-class report
    report_dict = classification_report(
        y_true, y_pred, target_names=label_names,
        zero_division=0, output_dict=True,
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        RESULTS_DIR / f"{tag}_per_class_metrics.csv"
    )

    # Summary
    summary = {
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
    }
    pd.DataFrame([summary]).to_csv(
        RESULTS_DIR / f"{tag}_eval_summary.csv", index=False
    )
    log.info("Results saved → %s/", RESULTS_DIR)
    return summary


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    model_dir: str,
) -> None:
    """Save a raw-count + row-normalised confusion matrix PNG."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    tag = Path(model_dir).name

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix — {tag}", fontsize=14, fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title("Raw Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=label_names, yticklabels=label_names,
                ax=axes[1], linewidths=0.5, vmin=0.0, vmax=1.0)
    axes[1].set_title("Row-Normalised (Recall per Class)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    out = RESULTS_DIR / f"{tag}_confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Confusion matrix plot saved → %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation(
    model_dir: str = DEFAULT_MODEL_DIR,
    debug: bool = False,
    force_cpu: bool = False,
    data_dir: str = None,
) -> None:
    """
    Full evaluation pipeline.

    Parameters
    ----------
    model_dir : str
        Directory with saved HuggingFace model + tokeniser (from train.py).
    debug : bool
        Use 99-row stratified sample if True.
    force_cpu : bool
        Force CPU — use when GPU driver is incompatible.
    data_dir : str | None
        Path to local CSV directory; falls back to HuggingFace Hub if None.
    """
    device = detect_device(force_cpu=force_cpu)
    y_true, y_pred, label_names, ms_per_sample = get_predictions(
        model_dir=model_dir, debug=debug, device=device, data_dir=data_dir
    )
    compute_and_log_metrics(y_true, y_pred, label_names, model_dir)
    # Append latency to the summary CSV
    tag = Path(model_dir).name
    import pandas as pd
    csv_path = RESULTS_DIR / f"{tag}_eval_summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["ms_per_sample"] = round(ms_per_sample, 3)
        df.to_csv(csv_path, index=False)
    log.info("Inference latency: %.2f ms / sample (appended to summary CSV)", ms_per_sample)
    plot_confusion_matrix(y_true, y_pred, label_names, model_dir)
    log.info("Evaluation complete. Results in '%s/'", RESULTS_DIR)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned model on MahaSent-MD."
    )
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help=f"Saved model directory (default: {DEFAULT_MODEL_DIR}).")
    parser.add_argument("--debug", action="store_true",
                        help="Evaluate on 99-row stratified sample only.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU execution.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to local CSV directory (skips HuggingFace Hub).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        model_dir=args.model_dir,
        debug=args.debug,
        force_cpu=args.cpu,
        data_dir=args.data_dir,
    )
