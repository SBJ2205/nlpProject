"""
data_loader.py
==============
Loads and tokenises the MahaSent-MD (L3Cube-MahaSent-MD) dataset.

Two data sources are supported — the script tries them in order:
1. **Local CSV files** (preferred) — set via --data_dir.
   This bypasses HuggingFace Hub entirely.
2. **HuggingFace Hub** (fallback) — requires a login token because the
   dataset is gated.  Set the HF_TOKEN env-var or run ``hf auth login``.

Usage
-----
# Local CSV mode (no HuggingFace login needed)
python src/data_loader.py --data_dir data/

# Debug mode — stratified 100-row sample per split
python src/data_loader.py --data_dir data/ --debug
"""

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_NAME: str = "l3cube-pune/marathi-sentiment-md"
# MuRIL is publicly available (no HuggingFace login needed).
# ai4bharat/indic-bert is gated — use --model ai4bharat/indic-bert after login.
DEFAULT_MODEL: str = "google/muril-base-cased"
MAX_LENGTH: int = 128
DEBUG_N_ROWS: int = 99          # divisible by 3 → exactly 33 per class
LABEL_COLUMN: str = "label"
TEXT_COLUMN: str = "text"

# Local CSV filenames (inside --data_dir)
LOCAL_CSV_TRAIN: str = "MahaSent_All_Train.csv"
LOCAL_CSV_VAL:   str = "MahaSent_All_Val.csv"
LOCAL_CSV_TEST:  str = "MahaSent_All_Test.csv"

# Label mapping: raw CSV integer → 0-indexed class id
LABEL_MAP: dict = {-1: 0, 0: 1, 1: 2}
LABEL_NAMES: list[str] = ["negative", "neutral", "positive"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalisation  (Objective 2)
# ---------------------------------------------------------------------------

# Zero-width and invisible Unicode characters common in social-media / OCR text
_ZW_CHARS = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\u00ad\ufeff\u2060\u2061\u2062\u2063]"
)
# URLs (http / https / www)
_URL_PAT = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)
# HTML entities and tags
_HTML_PAT = re.compile(r"<[^>]+>|&[a-z]+;")
# Runs of whitespace (tabs, newlines, multiple spaces) → single space
_WS_PAT = re.compile(r"[\t\n\r]+|[ ]{2,}")


def normalize_text(text: str) -> str:
    """
    Lightweight text normaliser for Marathi / code-mixed social-media text.

    Steps (Objective 2 of the project proposal):
      1. Remove URLs (http/https/www)
      2. Remove HTML tags and entities
      3. Strip zero-width and invisible Unicode characters
      4. Collapse runs of whitespace into a single space
      5. Strip leading / trailing whitespace

    Devanagari script and Unicode punctuation are intentionally preserved so
    that subword tokenisers (SentencePiece / WordPiece) can process them.

    Parameters
    ----------
    text : str

    Returns
    -------
    str  — normalised text
    """
    text = _URL_PAT.sub(" ", text)
    text = _HTML_PAT.sub(" ", text)
    text = _ZW_CHARS.sub("", text)
    text = _WS_PAT.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    """Read one split CSV and return a clean, normalised two-column DataFrame."""
    df = pd.read_csv(path, usecols=["text", "label"])
    # Apply text normalisation (Objective 2: strip URLs, zero-width chars, etc.)
    df["text"] = df["text"].fillna("").astype(str).apply(normalize_text)
    df["label"] = df["label"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def _df_to_dataset(df: pd.DataFrame) -> Dataset:
    """Convert a clean DataFrame to a typed HuggingFace Dataset."""
    features = Features({
        "text":  Value("string"),
        "label": ClassLabel(num_classes=3, names=LABEL_NAMES),
    })
    return Dataset.from_pandas(df, features=features, preserve_index=False)


# ---------------------------------------------------------------------------
# Stratified sampler
# ---------------------------------------------------------------------------

def _stratified_sample(dataset: Dataset, n: int) -> Dataset:
    """
    Return a stratified subsample of *dataset* with at most *n* rows.

    Uses ``dataset.select()`` — purely HuggingFace native, no pandas
    conversion — so there is no risk of dtype corruption.

    Parameters
    ----------
    dataset : Dataset
        Must contain an integer ``label`` column.
    n : int
        Target number of rows. The actual count may be slightly lower if
        one class has fewer rows than ``n // num_classes``.

    Returns
    -------
    Dataset
        Subset preserving the original feature schema.
    """
    labels = np.array(dataset[LABEL_COLUMN], dtype=np.int64)
    unique_classes = np.unique(labels)
    per_class = max(1, n // len(unique_classes))

    rng = np.random.default_rng(42)
    selected: list[int] = []
    for cls in unique_classes:
        indices = np.where(labels == cls)[0]
        k = min(per_class, len(indices))
        chosen = rng.choice(indices, k, replace=False)
        selected.extend(int(i) for i in chosen)

    return dataset.select(sorted(selected))


# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------

def load_raw_dataset(
    debug: bool = False,
    data_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Load the raw (un-tokenised) MahaSent-MD DatasetDict.

    Parameters
    ----------
    debug : bool
        When True, each split is reduced to ~``DEBUG_N_ROWS`` rows using
        stratified sampling so all three classes are always represented.
    data_dir : str | None
        Path to a local directory with the three CSV files.
        If None, falls back to HuggingFace Hub (requires auth token).

    Returns
    -------
    DatasetDict  with keys  ``train``, ``validation``, ``test``.
    """
    if data_dir:
        data_path = Path(data_dir)
        train_csv = data_path / LOCAL_CSV_TRAIN
        val_csv   = data_path / LOCAL_CSV_VAL
        test_csv  = data_path / LOCAL_CSV_TEST

        missing = [f for f in (train_csv, val_csv, test_csv) if not f.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing CSV files in '{}':\n  {}".format(
                    data_dir, "\n  ".join(str(m) for m in missing)
                )
                + "\n\nDownload them from GitHub (see README)."
            )

        log.info("Loading dataset from local CSVs in '%s' …", data_dir)
        ds = DatasetDict({
            "train":      _df_to_dataset(_load_csv(train_csv)),
            "validation": _df_to_dataset(_load_csv(val_csv)),
            "test":       _df_to_dataset(_load_csv(test_csv)),
        })

    else:
        from datasets import load_dataset
        log.info("Loading dataset '%s' from Hugging Face Hub …", DATASET_NAME)
        hf_token = os.environ.get("HF_TOKEN", None)
        if hf_token:
            log.info("Using HF_TOKEN env var for authentication.")
        else:
            log.info("No HF_TOKEN — relying on cached hf auth login.")
        ds = load_dataset(DATASET_NAME, token=hf_token)

    if debug:
        log.info(
            "DEBUG mode — stratified sample of ~%d rows per split.", DEBUG_N_ROWS
        )
        ds = DatasetDict(
            {split: _stratified_sample(ds[split], DEBUG_N_ROWS)
             for split in ds}
        )

    for split, subset in ds.items():
        log.info("  %-12s  %d rows", split, len(subset))

    return ds


def get_tokenised_datasets(
    model_name: str = DEFAULT_MODEL,
    debug: bool = False,
    max_length: int = MAX_LENGTH,
    data_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Load the raw dataset and apply the tokeniser for *model_name*.

    Returns
    -------
    DatasetDict  with columns  ``input_ids``, ``attention_mask``, ``labels``.
    """
    raw_ds = load_raw_dataset(debug=debug, data_dir=data_dir)

    log.info("Loading tokeniser for '%s' …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tokenise(batch: dict) -> dict:
        return tokenizer(
            batch[TEXT_COLUMN],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

    log.info("Tokenising dataset (max_length=%d) …", max_length)
    tok_ds = raw_ds.map(_tokenise, batched=True, batch_size=256, desc="Tokenising")

    # Drop the raw text column; rename label → labels for Trainer
    first_split = list(tok_ds.keys())[0]
    cols_to_drop = [c for c in [TEXT_COLUMN] if c in tok_ds[first_split].column_names]
    if cols_to_drop:
        tok_ds = tok_ds.remove_columns(cols_to_drop)

    if LABEL_COLUMN in tok_ds[first_split].column_names:
        tok_ds = tok_ds.rename_column(LABEL_COLUMN, "labels")

    tok_ds.set_format("torch")
    log.info("Tokenisation complete.")
    return tok_ds


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and tokenise the MahaSent-MD dataset."
    )
    parser.add_argument("--debug", action="store_true",
                        help="Stratified 99-row sample per split for quick tests.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Tokeniser model name (default: {DEFAULT_MODEL}).")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to local CSV directory (skips HuggingFace Hub).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ds = get_tokenised_datasets(
        model_name=args.model, debug=args.debug, data_dir=args.data_dir
    )
    print("\nFinal tokenised dataset:\n", ds)
