#!/usr/bin/env python
"""
Root entry point for Sarvam-1 Hardware-Adaptive QLoRA Fine-Tuning.

Usage:
    python train_sarvam.py --epochs 4
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from train_sarvam import _parse_args, train_sarvam

if __name__ == "__main__":
    args = _parse_args()
    train_sarvam(
        model_name=args.model,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        debug=args.debug,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        seq_length=args.seq_length,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        gradient_checkpointing=args.grad_checkpointing,
        force_cpu=args.cpu,
        skip_dry_run=args.no_dry_run,
    )
