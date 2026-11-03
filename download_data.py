"""
download_data.py
================
Downloads the official L3Cube-MahaSent-MD dataset files from GitHub:
  - MahaSent_All_Train.csv
  - MahaSent_All_Val.csv
  - MahaSent_All_Test.csv
into data/ directory.
"""

import urllib.request
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/l3cube-pune/MarathiNLP/main/L3Cube-MahaSent-MD/MahaSent_All"
FILES = [
    "MahaSent_All_Train.csv",
    "MahaSent_All_Val.csv",
    "MahaSent_All_Test.csv",
]

for filename in FILES:
    dest = DATA_DIR / filename
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[EXISTS] {filename} ({dest.stat().st_size:,} bytes)")
    else:
        url = f"{BASE_URL}/{filename}"
        print(f"[DOWNLOADING] {url} -> {dest} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"[DONE] {filename} ({dest.stat().st_size:,} bytes)")

print("\n" + "=" * 55)
print("DATASET SPLIT SIZES & CLASS DISTRIBUTION")
print("=" * 55)

LABEL_MAP = {-1: "negative", 0: "neutral", 1: "positive"}

for filename in FILES:
    df = pd.read_csv(DATA_DIR / filename)
    counts = df["label"].value_counts().to_dict()
    mapped = {LABEL_MAP.get(k, str(k)): counts.get(k, 0) for k in [-1, 0, 1]}
    print(f"{filename:<25} Total: {len(df):>6} rows  |  {mapped}")
