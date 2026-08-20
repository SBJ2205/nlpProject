#!/usr/bin/env python
"""
Root entry point for Sarvam-1 Interactive and Single-Sentence Sentiment Prediction.

Usage:
    python predict.py --text "हा चित्रपट खूप छान आहे"
    python predict.py --interactive
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict import SentimentPredictor, _format_result, _parse_args, run_interactive

if __name__ == "__main__":
    args = _parse_args()
    predictor = SentimentPredictor(model_dir=args.model_dir, force_cpu=args.cpu)

    if args.text:
        res = predictor.predict(args.text)
        print(_format_result(res))
    else:
        run_interactive(predictor)
