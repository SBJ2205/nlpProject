"""
domain_eval.py
==============
Cross-domain evaluation of a fine-tuned model against each of the four
L3Cube-MahaSent sub-datasets:

  - Movie Reviews   → l3cube-pune/marathi-sentiment-movie-reviews
  - Generic Tweets  → l3cube-pune/marathi-sentiment-tweets
  - TV Subtitles    → l3cube-pune/marathi-sentiment-subtitles
  - Political Tweets→ l3cube-pune/marathi-sentiment-political-tweets

Each dataset is downloaded from HuggingFace Hub on first run and cached
locally.  No HuggingFace login is required — all four are public.

The trained model must already exist in ``outputs/`` (run train.py first).
Results are saved to ``results/domain_eval/``.

Usage
-----
# Evaluate indic-bert checkpoint across all four domains (CPU)
python src/domain_eval.py --cpu

# Use a specific checkpoint
python src/domain_eval.py --model_dir outputs/google--muril-base-cased --cpu

# Quick sanity check (100 samples per domain)
python src/domain_eval.py --cpu --debug
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
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Make src/ importable when called from project root
sys.path.insert(0, str(Path(__file__).parent))
from train import detect_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR: str = "outputs/ai4bharat--indic-bert"
RESULTS_DIR: Path = Path("results") / "domain_eval"

LABEL_NAMES: list[str] = ["negative", "neutral", "positive"]
# Raw label values in all L3Cube domain CSVs/HF datasets: -1, 0, 1
LABEL_MAP: dict = {-1: 0, 0: 1, 1: 2}

# Domain display names, HuggingFace IDs (may be unavailable), and their
# position index (0-3) within the merged 'All' CSV file.
DOMAIN_CONFIGS: list[tuple[str, str, int]] = [
    ("Movie Reviews",    "l3cube-pune/marathi-sentiment-movie-reviews",    0),
    ("Generic Tweets",   "l3cube-pune/marathi-sentiment-tweets",            1),
    ("TV Subtitles",     "l3cube-pune/marathi-sentiment-subtitles",         2),
    ("Political Tweets", "l3cube-pune/marathi-sentiment-political-tweets",  3),
]

# Local CSV filenames (same as in data_loader.py)
LOCAL_CSV_TEST:  str = "MahaSent_All_Test.csv"
LOCAL_CSV_TRAIN: str = "MahaSent_All_Train.csv"

BATCH_SIZE: int = 64
DEBUG_N: int = 100


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _normalise_dataset(raw) -> Dataset:
    """
    Normalise a raw HuggingFace Dataset into a two-column Dataset with
    columns ``text`` (str) and ``label`` (0-indexed int: 0=neg, 1=neu, 2=pos).

    Handles two observed label formats from L3Cube:
      - Integer ClassLabel already mapped to 0/1/2
      - String label "-1" / "0" / "1"  (raw CSV upload)
    """
    texts = []
    labels = []

    for row in raw:
        # Identify the text column
        text_col = next(
            (c for c in ["text", "sentence", "review", "content"] if c in row),
            None,
        )
        label_col = next(
            (c for c in ["label", "sentiment", "class"] if c in row),
            None,
        )
        if text_col is None or label_col is None:
            continue

        raw_label = row[label_col]
        # If the dataset already uses 0/1/2 (ClassLabel with names=[neg,neu,pos])
        # keep as-is; if it uses -1/0/1 integers or strings, map them.
        if isinstance(raw_label, str):
            try:
                raw_label = int(raw_label)
            except ValueError:
                raw_label = {"negative": -1, "neutral": 0, "positive": 1}.get(
                    raw_label.lower(), 0
                )

        if isinstance(raw_label, (int, np.integer)):
            if raw_label in LABEL_MAP:
                mapped = LABEL_MAP[int(raw_label)]
            else:
                # Already 0-indexed
                mapped = int(raw_label)
        else:
            mapped = int(raw_label)

        texts.append(str(row[text_col]).strip())
        labels.append(mapped)

    features = Features({
        "text":  Value("string"),
        "label": ClassLabel(num_classes=3, names=LABEL_NAMES),
    })
    df = pd.DataFrame({"text": texts, "label": labels})
    return Dataset.from_pandas(df, features=features, preserve_index=False)


def load_domain_dataset(
    hf_id: str,
    domain_idx: int,
    debug: bool = False,
    data_dir: str = None,
) -> Dataset:
    """
    Load one domain's test data, with a two-level fallback strategy:

    1. **HuggingFace Hub** — tries ``hf_id`` first (requires connectivity).
    2. **Local CSV split** — if HF download fails, splits the user's existing
       ``MahaSent_All_Test.csv`` into 4 equal consecutive chunks and returns
       the chunk at index ``domain_idx`` (0=movie, 1=tweets, 2=subtitles,
       3=political).  This works because the 'All' file is a concatenation
       of the four domain files in that fixed order.

    In debug mode returns at most ``DEBUG_N`` rows from whichever source
    was loaded.
    """
    # ── Attempt 1: HuggingFace Hub ───────────────────────────────────────────
    from datasets import load_dataset

    log.info("  Trying HuggingFace Hub: '%s' …", hf_id)
    try:
        ds_dict = load_dataset(hf_id)
        split = "test" if "test" in ds_dict else list(ds_dict.keys())[0]
        raw = ds_dict[split]
        log.info("  ✓ HuggingFace — split '%s'  (%d rows)", split, len(raw))
        normalised = _normalise_dataset(raw)

    except Exception as hf_exc:
        log.warning("  HuggingFace unavailable (%s)", hf_exc)

        # ── Attempt 2: local CSV split ────────────────────────────────────────
        normalised = _load_from_local_csv_split(
            domain_idx=domain_idx, data_dir=data_dir
        )

    # ── Subsample in debug mode ──────────────────────────────────────────────
    if debug and len(normalised) > DEBUG_N:
        rng = np.random.default_rng(42)
        idx = sorted(
            rng.choice(len(normalised), DEBUG_N, replace=False).tolist()
        )
        normalised = normalised.select([int(i) for i in idx])
        log.info("  Debug: subsampled to %d rows", len(normalised))

    return normalised


def _load_from_local_csv_split(
    domain_idx: int,
    data_dir: str = None,
) -> Dataset:
    """
    Split the local MahaSent_All_Test.csv into 4 equal consecutive chunks
    and return the chunk at *domain_idx*.

    The MahaSent-MD 'All' file is a concatenation of the four domain CSVs
    in the order: movie (0), tweets (1), subtitles (2), political (3).

    Parameters
    ----------
    domain_idx : int  in 0..3
    data_dir   : str | None
        Directory containing the CSV.  Defaults to ``data/`` relative to CWD.
    """
    import csv

    csv_dir  = Path(data_dir) if data_dir else Path("data")
    csv_path = csv_dir / LOCAL_CSV_TEST

    if not csv_path.exists():
        # Try train split as a last resort
        csv_path = csv_dir / LOCAL_CSV_TRAIN

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cannot find local CSV for fallback domain split.\n"
            f"Expected one of:\n"
            f"  {csv_dir / LOCAL_CSV_TEST}\n"
            f"  {csv_dir / LOCAL_CSV_TRAIN}\n"
            f"Run the project from its root directory or pass --data_dir."
        )

    log.info("  ✓ Local CSV fallback: splitting '%s' …", csv_path.name)

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n_total  = len(rows)
    chunk    = n_total // 4
    start    = domain_idx * chunk
    end      = start + chunk if domain_idx < 3 else n_total  # last chunk gets remainder
    chunk_rows = rows[start:end]

    texts  = []
    labels = []
    for row in chunk_rows:
        text  = str(row.get("text", "")).strip()
        try:
            raw_label = int(row.get("label", 0))
        except (ValueError, TypeError):
            raw_label = 0
        mapped = LABEL_MAP.get(raw_label, raw_label)
        texts.append(text)
        labels.append(mapped)

    log.info(
        "  Rows %d–%d of %d  (%d rows for this domain)",
        start, end - 1, n_total, len(texts),
    )

    features = Features({
        "text":  Value("string"),
        "label": ClassLabel(num_classes=3, names=LABEL_NAMES),
    })
    df = pd.DataFrame({"text": texts, "label": labels})
    return Dataset.from_pandas(df, features=features, preserve_index=False)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run inference on *dataset*.

    Returns
    -------
    (y_true, y_pred, ms_per_sample)
        ``y_true``        — ground-truth integer labels
        ``y_pred``        — predicted integer labels
        ``ms_per_sample`` — average inference time in milliseconds
    """
    texts: list[str] = dataset["text"]
    y_true: np.ndarray = np.array(dataset["label"], dtype=np.int64)

    all_preds: list[int] = []
    total_time: float = 0.0
    n_samples: int = 0

    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i: i + BATCH_SIZE]
            enc = tokenizer(
                batch,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            t0 = time.perf_counter()
            logits = model(**enc).logits
            t1 = time.perf_counter()

            total_time += (t1 - t0)
            n_samples += len(batch)
            all_preds.extend(
                torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            )

    ms_per_sample = (total_time / n_samples) * 1000 if n_samples > 0 else 0.0
    return y_true, np.array(all_preds, dtype=np.int64), ms_per_sample


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    domain_name: str,
) -> dict:
    """Return a dict of macro-averaged metrics for one domain."""
    return {
        "domain": domain_name,
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "n_samples": int(len(y_true)),
    }


def print_domain_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    domain_name: str,
    ms_per_sample: float,
) -> None:
    """Pretty-print per-class classification report for one domain."""
    log.info("=" * 58)
    log.info("Domain : %s", domain_name)
    log.info("Latency: %.2f ms / sample", ms_per_sample)
    log.info("-" * 58)
    log.info(
        "\n%s",
        classification_report(
            y_true, y_pred,
            target_names=LABEL_NAMES,
            zero_division=0,
        ),
    )


def plot_comparison(summary_df: pd.DataFrame, model_tag: str) -> None:
    """
    Generate a grouped bar chart comparing Macro F1 / Precision / Recall
    across all four domains and save it to RESULTS_DIR.
    """
    metrics = ["macro_f1", "macro_precision", "macro_recall"]
    metric_labels = ["Macro F1", "Macro Precision", "Macro Recall"]

    n_domains = len(summary_df)
    n_metrics = len(metrics)
    x = np.arange(n_domains)
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        f"Cross-Domain Sentiment Performance — {model_tag}",
        fontsize=14,
        fontweight="bold",
    )

    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    for i, (metric, label, color) in enumerate(
        zip(metrics, metric_labels, colors)
    ):
        values = summary_df[metric].tolist()
        bars = ax.bar(
            x + i * width,
            values,
            width,
            label=label,
            color=color,
            alpha=0.85,
            edgecolor="white",
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.005,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Domain", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(summary_df["domain"].tolist(), fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = RESULTS_DIR / f"{model_tag}_domain_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Bar chart saved → %s", out)


def plot_latency(summary_df: pd.DataFrame, model_tag: str) -> None:
    """Save a horizontal bar chart of inference latency per domain."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(
        summary_df["domain"],
        summary_df["ms_per_sample"],
        color="#9C27B0",
        alpha=0.8,
    )
    for i, (val, domain) in enumerate(
        zip(summary_df["ms_per_sample"], summary_df["domain"])
    ):
        ax.text(val + 0.02, i, f"{val:.2f} ms", va="center", fontsize=9)
    ax.set_xlabel("Avg Inference Latency (ms / sample)", fontsize=11)
    ax.set_title(
        f"Inference Latency by Domain — {model_tag}", fontsize=13, fontweight="bold"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = RESULTS_DIR / f"{model_tag}_latency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Latency chart saved → %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_domain_evaluation(
    model_dir: str = DEFAULT_MODEL_DIR,
    debug: bool = False,
    force_cpu: bool = False,
    data_dir: str = None,
) -> pd.DataFrame:
    """
    Full cross-domain evaluation pipeline.

    Parameters
    ----------
    model_dir : str
        Directory with saved HuggingFace model + tokeniser (from train.py).
    debug : bool
        Subsample each domain to 100 rows for a quick sanity-check.
    force_cpu : bool
        Skip GPU — use when CUDA is not available.
    data_dir : str | None
        Path to local CSV directory for the fallback domain split
        (defaults to ``data/`` relative to CWD).

    Returns
    -------
    pd.DataFrame
        Summary table with one row per domain.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = detect_device(force_cpu=force_cpu)

    # Resolve to an absolute Path object.
    # Use local_files_only=True so transformers never tries to validate
    # the path as a HuggingFace repo ID (Windows paths with '\' and '--'
    # fail HF's repo-id validation even when the directory exists locally).
    model_path = Path(model_dir).resolve()

    if not model_path.exists():
        available = sorted(
            p.name for p in Path("outputs").iterdir() if p.is_dir()
        ) if Path("outputs").exists() else []
        hint = (
            f"\n  Available checkpoints in outputs/:\n"
            + "\n".join(f"    --model_dir outputs/{n}" for n in available)
            if available else ""
        )
        raise FileNotFoundError(
            f"Model directory not found: '{model_path}'\n"
            f"Run 'python src/train.py --cpu --debug' first to create a checkpoint.{hint}"
        )

    log.info("Loading model from '%s' …", model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )
    model.to(device)

    # Model size
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "Model parameters: %s  (%.1f M)",
        f"{n_params:,}",
        n_params / 1e6,
    )

    model_tag = Path(model_dir).name
    all_metrics: list[dict] = []

    for domain_name, hf_id, domain_idx in DOMAIN_CONFIGS:
        log.info("\n── %s ──", domain_name)
        try:
            dataset = load_domain_dataset(
                hf_id=hf_id,
                domain_idx=domain_idx,
                debug=debug,
                data_dir=data_dir,
            )
        except Exception as exc:
            log.error("  Could not load '%s': %s", domain_name, exc)
            log.warning("  Skipping '%s'.", domain_name)
            continue

        y_true, y_pred, ms_per_sample = run_inference(
            model, tokenizer, dataset, device
        )
        print_domain_report(y_true, y_pred, domain_name, ms_per_sample)

        row = compute_metrics(y_true, y_pred, domain_name)
        row["ms_per_sample"] = round(ms_per_sample, 3)
        all_metrics.append(row)

    if not all_metrics:
        log.error("No domain results collected — check HuggingFace connectivity.")
        return pd.DataFrame()

    summary_df = pd.DataFrame(all_metrics)

    # Console summary table
    log.info("\n%s", "=" * 58)
    log.info("CROSS-DOMAIN SUMMARY")
    log.info("%s", "=" * 58)
    log.info(
        "\n%s",
        summary_df[
            ["domain", "macro_f1", "macro_precision", "macro_recall",
             "ms_per_sample", "n_samples"]
        ].to_string(index=False)
    )

    # Save CSV
    csv_path = RESULTS_DIR / f"{model_tag}_domain_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    log.info("Summary CSV saved → %s", csv_path)

    # Plots
    plot_comparison(summary_df, model_tag)
    plot_latency(summary_df, model_tag)

    log.info(
        "\nDone. All results saved to '%s/'", RESULTS_DIR
    )
    return summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-domain evaluation of a fine-tuned Marathi sentiment model "
            "across all four L3Cube-MahaSent sub-datasets."
        )
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Saved model directory (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Subsample each domain to 100 rows for a quick test.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Path to local CSV directory for the fallback domain split "
            "(default: data/). Used when HuggingFace Hub is unavailable."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_domain_evaluation(
        model_dir=args.model_dir,
        debug=args.debug,
        force_cpu=args.cpu,
        data_dir=args.data_dir,
    )
