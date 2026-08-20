#!/usr/bin/env python
"""
Root entry point for Sarvam-1 Evaluation and Baseline Comparison.

Usage:
    python evaluate_sarvam.py
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluate_sarvam import _parse_args, evaluate_sarvam

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
