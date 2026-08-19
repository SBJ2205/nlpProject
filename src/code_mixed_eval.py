"""
code_mixed_eval.py
==================
Qualitative evaluation of the fine-tuned model on a curated held-out set of
**code-mixed (Romanized Marathi / Hindi-English mix) social-media text**.

This fulfils the "small held-out set of code-mixed text" requirement from the
project proposal (Page 11).  Since no publicly annotated code-mixed Marathi
dataset is freely available, a representative set of 30 hand-curated sentences
is embedded directly in this script.  Sentences span all three sentiment classes
and cover common code-mixing patterns observed on social media.

Usage
-----
python src/code_mixed_eval.py --cpu
python src/code_mixed_eval.py --cpu --model_dir outputs/google--muril-base-cased
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from predict import SentimentPredictor
from train import detect_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "outputs/ai4bharat--indic-bert"
RESULTS_DIR = Path("results") / "code_mixed_eval"
LABEL_NAMES = ["negative", "neutral", "positive"]

# ---------------------------------------------------------------------------
# Curated code-mixed held-out set  (30 sentences)
# ---------------------------------------------------------------------------
# Format: (text, gold_label_index)
#   0 = negative, 1 = neutral, 2 = positive
#
# Patterns covered:
#   - Romanized Marathi  (e.g. "khup chan" = very good)
#   - Marathi-English mix  (e.g. "movie khup boring aahe")
#   - Hindi-Marathi mix  (e.g. "bahut accha aahe")
#   - Social-media abbreviations  (LOL, TBH, idk)
# ---------------------------------------------------------------------------

CODE_MIXED_SAMPLES: list[tuple[str, int]] = [
    # --- Positive (label 2) ---
    ("khup chan movie aahe, must watch!",                          2),
    ("aaj khup chand din gela, totally loved it",                  2),
    ("he gaane ekdum best aahe yaar",                              2),
    ("actor cha performance ekdum top class aahe",                 2),
    ("majha favorite show ahe, best episode ever!",                2),
    ("bahut accha aahe bhai, truly amazing story",                 2),
    ("this product is genuinely khup chan, 5 stars!",              2),
    ("game khup addictive aahe, non stop khelto me",               2),
    ("teacher ne khup chan samjavle, really helpful",               2),
    ("mala khup aanand jhala, best day of my life",                2),
    # --- Neutral (label 1) ---
    ("thik aahe, na too good na too bad",                          1),
    ("movie average aahe, time pass chhan ahe",                    1),
    ("product okay aahe, as expected",                             1),
    ("majha opinion different aahe, each to their own",            1),
    ("TBH mala mahit nahi, idk about this",                        1),
    ("normal aahe, nothing special about it",                      1),
    ("he news just for information aahe, no views",                1),
    ("situation thodi complicated aahe, will see",                 1),
    ("mixed feelings aahet, can't decide",                         1),
    ("aaj weather thik aahe, neither hot nor cold",                1),
    # --- Negative (label 0) ---
    ("khup boring movie aahe, waste of time",                      0),
    ("he product ekdum bakwaas aahe, don't buy",                   0),
    ("service khup poori aahe, very disappointed",                 0),
    ("mala khup raga ala, worst experience ever",                  0),
    ("actor cha performance terrible aahe yaar, LOL no",           0),
    ("bahut bura laga, really hurt ho gaya",                       0),
    ("he gaana mala bilkul avdla nahi, waste",                     0),
    ("traffic khup jam aahe, irritating day",                      0),
    ("exam khup tough hota, failed re",                            0),
    ("he restaurant cha food ekdum kharab aahe, never again",      0),
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_code_mixed_evaluation(
    model_dir: str = DEFAULT_MODEL_DIR,
    force_cpu: bool = False,
) -> pd.DataFrame:
    """
    Evaluate the fine-tuned model on the built-in code-mixed sample set.

    Returns a DataFrame with per-sentence predictions and gold labels.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = detect_device(force_cpu=force_cpu)
    predictor = SentimentPredictor(model_dir=model_dir, device=device)
    model_tag = Path(model_dir).name
    if "sarvam" in model_tag.lower():
        model_tag = "sarvam-1"

    texts      = [s for s, _ in CODE_MIXED_SAMPLES]
    gold_labels = [lbl for _, lbl in CODE_MIXED_SAMPLES]

    log.info("Running inference on %d code-mixed sentences …", len(texts))
    results   = predictor.predict_batch(texts)
    pred_labels = [LABEL_NAMES.index(r["label"]) for r in results]
    confidences = [r["confidence"] for r in results]
    latencies   = [r["latency_ms"] for r in results]

    # ---------- Console report ----------
    macro_f1 = f1_score(gold_labels, pred_labels, average="macro", zero_division=0)
    log.info("=" * 58)
    log.info("CODE-MIXED EVALUATION RESULTS")
    log.info("=" * 58)
    log.info("  Model     : %s", model_tag)
    log.info("  Sentences : %d", len(texts))
    log.info("  Macro F1  : %.4f", macro_f1)
    log.info("  Avg Latency: %.2f ms / sample", float(np.mean(latencies)))
    log.info("-" * 58)
    log.info(
        "\n%s",
        classification_report(
            gold_labels, pred_labels,
            target_names=LABEL_NAMES,
            zero_division=0,
        ),
    )

    # ---------- Per-sentence table ----------
    df = pd.DataFrame({
        "text":        texts,
        "gold":        [LABEL_NAMES[l] for l in gold_labels],
        "predicted":   [LABEL_NAMES[l] for l in pred_labels],
        "correct":     [g == p for g, p in zip(gold_labels, pred_labels)],
        "confidence":  [round(c, 4) for c in confidences],
        "latency_ms":  [round(l, 2) for l in latencies],
    })
    csv_path = RESULTS_DIR / f"{model_tag}_code_mixed_results.csv"
    df.to_csv(csv_path, index=False)
    log.info("Per-sentence results saved → %s", csv_path)

    # ---------- Confusion matrix ----------
    cm = confusion_matrix(gold_labels, pred_labels)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Code-Mixed Evaluation — {model_tag}", fontsize=13, fontweight="bold")

    import seaborn as sns
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title("Raw Counts")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title("Row-Normalised")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    plt.tight_layout()
    cm_path = RESULTS_DIR / f"{model_tag}_code_mixed_confusion.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Confusion matrix saved → %s", cm_path)

    log.info("Done. All results in '%s/'", RESULTS_DIR)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model on code-mixed Marathi/Hindi sentences."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Saved model directory (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_code_mixed_evaluation(
        model_dir=args.model_dir,
        force_cpu=args.cpu,
    )
