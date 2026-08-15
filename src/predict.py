"""
predict.py
==========
Interactive real-time sentiment prediction on custom Marathi / code-mixed text.
(Objective: lightweight inference script — Project Proposal Page 10 & 16)

Supports two modes:
  1. Interactive REPL  — type sentences one at a time, get live predictions.
  2. Batch mode        — pass --text "some sentence" for single prediction,
                         or --file sentences.txt for bulk prediction.

Usage
-----
# Interactive mode (start typing Marathi text)
python src/predict.py --cpu

# Single sentence
python src/predict.py --cpu --text "हा चित्रपट खूप छान आहे"

# Bulk prediction from a text file (one sentence per line)
python src/predict.py --cpu --file my_sentences.txt

# Use a different checkpoint
python src/predict.py --cpu --model_dir outputs/google--muril-base-cased
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Make src/ importable when called from project root
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import normalize_text
from train import detect_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_MODEL_DIR: str = "outputs/ai4bharat--indic-bert"
LABEL_NAMES: list[str] = ["negative", "neutral", "positive"]
LABEL_EMOJI: dict[str, str] = {
    "negative": "[--] Negative",
    "neutral":  "[~~] Neutral",
    "positive": "[++] Positive",
}


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class SentimentPredictor:
    """
    Wraps a fine-tuned HuggingFace model for single- and batch-sentence
    Marathi sentiment inference.

    Parameters
    ----------
    model_dir : str
        Local directory with a saved model + tokeniser (from train.py).
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(self, model_dir: str, device: str = "cpu") -> None:
        model_path = Path(model_dir).resolve()
        if not model_path.exists():
            available = sorted(
                p.name for p in Path("outputs").iterdir() if p.is_dir()
            ) if Path("outputs").exists() else []
            hint = (
                "\n  Available checkpoints:\n" +
                "\n".join(f"    --model_dir outputs/{n}" for n in available)
                if available else ""
            )
            raise FileNotFoundError(
                f"Model directory not found: '{model_path}'\n"
                f"Run 'python src/train.py --cpu --debug' to create one.{hint}"
            )

        log.info("Loading model from '%s' …", model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.device = device
        self.model.to(device)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        log.info("Model loaded  — %s parameters (%.1f M)", f"{n_params:,}", n_params / 1e6)
        log.info("Device        — %s", device.upper())

    def predict(self, text: str) -> dict:
        """
        Predict sentiment for a single text string.

        Parameters
        ----------
        text : str
            Raw Marathi or code-mixed input.

        Returns
        -------
        dict with keys:
            ``label``       — predicted class name (str)
            ``confidence``  — softmax probability of the predicted class (float)
            ``scores``      — dict mapping each class name to its probability
            ``latency_ms``  — inference time in milliseconds (float)
            ``input``       — normalised input text (str)
        """
        normalised = normalize_text(text)
        enc = self.tokenizer(
            normalised,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(**enc).logits
        latency_ms = (time.perf_counter() - t0) * 1000

        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        if not isinstance(probs, list):
            probs = [probs]

        pred_idx = int(torch.argmax(logits, dim=-1).item())
        return {
            "label":      LABEL_NAMES[pred_idx],
            "confidence": probs[pred_idx],
            "scores":     dict(zip(LABEL_NAMES, probs)),
            "latency_ms": latency_ms,
            "input":      normalised,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Run predict() on a list of sentences."""
        return [self.predict(t) for t in texts]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _format_result(result: dict) -> str:
    label    = result["label"]
    emoji    = LABEL_EMOJI.get(label, label)
    conf_pct = result["confidence"] * 100
    filled   = int(conf_pct // 5)
    bar      = "=" * filled + "-" * (20 - filled)
    lines = [
        "",
        f"  Prediction  : {emoji}  ({conf_pct:.1f}% confidence)",
        f"  Confidence  : [{bar}]",
        f"  Latency     : {result['latency_ms']:.1f} ms",
        "  Scores      :",
    ]
    for cls, prob in result["scores"].items():
        marker = " <--" if cls == label else ""
        lines.append(f"    {cls:<10} {prob * 100:>5.1f}%{marker}")
    lines.append(f"  Input text  : {result['input'][:80]}{'...' if len(result['input']) > 80 else ''}")
    return "\n".join(lines)


def _print_header(model_dir: str) -> None:
    print("\n" + "═" * 58)
    print("  Marathi Sentiment Predictor")
    print(f"  Model : {Path(model_dir).name}")
    print("  Type a sentence and press Enter. Type 'quit' to exit.")
    print("═" * 58)


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_interactive(predictor: SentimentPredictor, model_dir: str) -> None:
    """Start the interactive REPL loop."""
    _print_header(model_dir)
    while True:
        try:
            text = input("\n▶  Enter text: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not text:
            continue
        if text.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        result = predictor.predict(text)
        print(_format_result(result))


def run_single(predictor: SentimentPredictor, text: str) -> None:
    """Predict and print for a single --text argument."""
    result = predictor.predict(text)
    print(_format_result(result))


def run_file(predictor: SentimentPredictor, filepath: str) -> None:
    """Predict all sentences in a text file (one per line)."""
    path = Path(filepath)
    if not path.exists():
        log.error("File not found: %s", filepath)
        sys.exit(1)

    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    log.info("Running batch prediction on %d sentences from '%s' …", len(lines), filepath)
    print("\n" + "═" * 58)
    for i, sentence in enumerate(lines, 1):
        result = predictor.predict(sentence)
        print(f"\n[{i}/{len(lines)}]{_format_result(result)}")
    print("\n" + "═" * 58)
    log.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive Marathi sentiment predictor (Project Proposal Objective)."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Saved model directory (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single sentence to predict (skips interactive mode).",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a .txt file with one sentence per line for batch prediction.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    device = detect_device(force_cpu=args.cpu)
    predictor = SentimentPredictor(model_dir=args.model_dir, device=device)

    if args.file:
        run_file(predictor, args.file)
    elif args.text:
        run_single(predictor, args.text)
    else:
        run_interactive(predictor, args.model_dir)
