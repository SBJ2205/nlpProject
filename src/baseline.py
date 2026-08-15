"""
baseline.py
===========
Classical TF-IDF + Logistic Regression baseline for the
MahaSent-MD Marathi sentiment dataset.

Provides a fast, interpretable reference point against which the
fine-tuned transformer models are compared.

Usage
-----
python src/baseline.py --data_dir data/ --debug   # 99-row stratified sample
python src/baseline.py --data_dir data/            # full dataset
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

# Make src/ importable when called from project root
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import LABEL_COLUMN, TEXT_COLUMN, load_raw_dataset

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    Build a TF-IDF + Logistic Regression sklearn Pipeline.

    Character-level n-grams (2–4) improve coverage for morphologically
    rich languages like Marathi and code-mixed text.

    Notes
    -----
    ``multi_class`` was removed in scikit-learn 1.5 — the ``lbfgs`` solver
    handles multi-class natively via softmax, so no extra flag is needed.
    ``n_jobs`` is not supported by ``lbfgs`` and is intentionally omitted.
    """
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",      # Character-level n-grams
        ngram_range=(2, 4),
        max_features=50_000,
        sublinear_tf=True,       # log(1 + tf) normalisation
        strip_accents=None,      # Preserve Devanagari / Unicode chars
        min_df=2,
    )
    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict:
    """Compute Macro F1 / Precision / Recall, print them, return as dict."""
    macro_f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)

    log.info("=" * 55)
    log.info("BASELINE RESULTS  (TF-IDF + Logistic Regression)")
    log.info("=" * 55)
    log.info("  Macro F1        : %.4f", macro_f1)
    log.info("  Macro Precision : %.4f", macro_prec)
    log.info("  Macro Recall    : %.4f", macro_rec)
    log.info("-" * 55)
    log.info(
        "\n%s",
        classification_report(y_true, y_pred, target_names=label_names,
                               zero_division=0),
    )
    return {"macro_f1": macro_f1, "macro_precision": macro_prec,
            "macro_recall": macro_rec}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_baseline(debug: bool = False, data_dir: str = None) -> None:
    """
    End-to-end baseline: load data → fit → evaluate → save results.

    Parameters
    ----------
    debug : bool
        Use stratified 99-row sample per split if True.
    data_dir : str | None
        Path to local CSV directory; falls back to HuggingFace Hub if None.
    """
    # 1. Load raw (string) dataset
    ds = load_raw_dataset(debug=debug, data_dir=data_dir)

    train_split = ds["train"]
    eval_key    = "validation" if "validation" in ds else "test"
    eval_split  = ds[eval_key]

    # Extract text lists and integer label arrays
    X_train: list[str] = train_split[TEXT_COLUMN]
    X_eval:  list[str] = eval_split[TEXT_COLUMN]
    y_train = np.array(train_split[LABEL_COLUMN], dtype=np.int64)
    y_eval  = np.array(eval_split[LABEL_COLUMN],  dtype=np.int64)

    # Derive label names from ClassLabel feature (or fallback to string ids)
    if hasattr(train_split.features[LABEL_COLUMN], "names"):
        label_names: list[str] = train_split.features[LABEL_COLUMN].names
    else:
        label_names = [str(c) for c in sorted(set(y_train.tolist()))]

    log.info("Train samples : %d", len(X_train))
    log.info("Eval  samples : %d", len(X_eval))
    log.info("Label classes : %s", label_names)
    log.info("Label dist (train): %s", dict(zip(*np.unique(y_train, return_counts=True))))

    # 2. Fit
    log.info("Fitting TF-IDF + Logistic Regression …")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # 3. Predict & evaluate
    y_pred = pipe.predict(X_eval)
    metrics = evaluate_predictions(y_eval, y_pred, label_names)

    # 4. Save results
    results_path = RESULTS_DIR / "baseline_metrics.csv"
    pd.DataFrame([metrics]).to_csv(results_path, index=False)
    log.info("Metrics saved → %s", results_path)

    cm = confusion_matrix(y_eval, y_pred)
    cm_path = RESULTS_DIR / "baseline_confusion_matrix.txt"
    np.savetxt(cm_path, cm, fmt="%d", header=" ".join(label_names))
    log.info("Confusion matrix saved → %s", cm_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TF-IDF + Logistic Regression baseline for Marathi Sentiment."
    )
    parser.add_argument("--debug", action="store_true",
                        help="Stratified 99-row sample per split for quick testing.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to local CSV directory (skips HuggingFace Hub).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_baseline(debug=args.debug, data_dir=args.data_dir)
