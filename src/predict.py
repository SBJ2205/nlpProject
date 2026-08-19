"""
predict.py
==========
Interactive real-time sentiment prediction on custom Marathi / code-mixed text.
Supports both encoder-based classification models (IndicBERT, MuRIL) and
generative Small Language Models (Sarvam-1 with LoRA adapters).

Supports two modes:
  1. Interactive REPL  — type sentences one at a time, get live predictions.
  2. Batch mode        — pass --text "some sentence" for single prediction,
                         or --file sentences.txt for bulk prediction.

Usage
-----
# Interactive mode with default model
python src/predict.py --cpu

# Single sentence with Sarvam-1 LoRA adapter
python src/predict.py --model_dir saved_models/sarvam-1-lora --text "हा चित्रपट खूप छान आहे"

# Bulk prediction from a text file
python src/predict.py --file my_sentences.txt
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Make src/ importable when called from project root
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import normalize_text
from train import detect_device

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

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

PROMPT_TEMPLATE: str = (
    "Classify the sentiment of the following Marathi text. Reply with ONLY ONE WORD from [Positive, Negative, Neutral].\n"
    "Text: {text_sample}\n"
    "Sentiment:"
)


def parse_generated_sentiment(text: str) -> tuple[int, str]:
    """Parse raw generation into label index and name."""
    cleaned = text.strip().lower()
    tokens = re.findall(r"\b\w+\b", cleaned)
    first_token = tokens[0] if tokens else cleaned

    if "pos" in first_token or "आवड" in cleaned or "chan" in cleaned:
        return 2, "positive"
    elif "neg" in first_token or "वाईट" in cleaned or "bakwaas" in cleaned:
        return 0, "negative"
    elif "neu" in first_token or "मध्य" in cleaned or "thik" in cleaned:
        return 1, "neutral"

    if "positive" in cleaned:
        return 2, "positive"
    if "negative" in cleaned:
        return 0, "negative"
    if "neutral" in cleaned:
        return 1, "neutral"

    return 1, "neutral"


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class SentimentPredictor:
    """
    Wraps fine-tuned HuggingFace encoder models or generative SLMs (Sarvam-1)
    for single- and batch-sentence Marathi sentiment inference.
    """

    def __init__(self, model_dir: str, device: str = "cpu") -> None:
        model_path = Path(model_dir).resolve()
        self.device = device
        self.is_generative = False

        if not model_path.exists():
            # Check if it's a HuggingFace hub id
            model_path_str = model_dir
        else:
            model_path_str = str(model_path)

        # Detect whether it is a LoRA adapter or Causal LM
        is_adapter = (Path(model_path_str) / "adapter_config.json").exists() if Path(model_path_str).exists() else False
        is_sarvam = "sarvam" in model_path_str.lower()

        if is_adapter or is_sarvam:
            self.is_generative = True
            self._init_generative_model(model_path_str, is_adapter)
        else:
            self._init_classification_model(model_path_str)

    def _init_classification_model(self, model_path_str: str) -> None:
        """Initialize encoder-based sequence classification model."""
        log.info("Loading sequence classification model from '%s' …", model_path_str)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path_str, local_files_only=Path(model_path_str).exists()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path_str, local_files_only=Path(model_path_str).exists()
        )
        self.model.to(self.device)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        log.info("Model loaded  — %s parameters (%.1f M)", f"{n_params:,}", n_params / 1e6)
        log.info("Device        — %s", self.device.upper())

    def _init_generative_model(self, model_path_str: str, is_adapter: bool) -> None:
        """Initialize generative Causal LM with optional LoRA adapter."""
        base_model_name = "sarvamai/sarvam-1"
        if is_adapter:
            try:
                with open(Path(model_path_str) / "adapter_config.json", encoding="utf-8") as f:
                    cfg = json.load(f)
                    base_model_name = cfg.get("base_model_name_or_path", base_model_name)
            except Exception:
                pass

        use_cuda = (self.device == "cuda" or (torch.cuda.is_available() and self.device != "cpu"))
        log.info("Loading generative tokenizer for '%s' …", base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        log.info("Loading base causal model '%s' (CUDA=%s) …", base_model_name, use_cuda)
        if use_cuda:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

        if is_adapter and PeftModel is not None:
            log.info("Attaching LoRA adapter from '%s' …", model_path_str)
            self.model = PeftModel.from_pretrained(base_model, model_path_str)
        else:
            self.model = base_model

        self.model.eval()
        log.info("Generative SLM initialized on device '%s'.", self.device)

    def predict(self, text: str) -> dict:
        """
        Predict sentiment for a single text string.
        """
        normalised = normalize_text(text)

        if self.is_generative:
            prompt = PROMPT_TEMPLATE.format(text_sample=normalised)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device if hasattr(self.model, "device") else self.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            latency_ms = (time.perf_counter() - t0) * 1000

            new_tokens = outputs[0][input_len:]
            gen_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            pred_idx, pred_label = parse_generated_sentiment(gen_text)

            # Heuristic distribution for generative single-token output
            scores = {name: (0.90 if i == pred_idx else 0.05) for i, name in enumerate(LABEL_NAMES)}
            confidence = scores[pred_label]

            return {
                "label": pred_label,
                "confidence": confidence,
                "scores": scores,
                "latency_ms": latency_ms,
                "input": normalised,
                "raw_generation": gen_text,
            }

        # Sequence classification encoder pipeline
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
            "label": LABEL_NAMES[pred_idx],
            "confidence": probs[pred_idx],
            "scores": dict(zip(LABEL_NAMES, probs)),
            "latency_ms": latency_ms,
            "input": normalised,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Run predict() on a list of sentences."""
        return [self.predict(t) for t in texts]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _format_result(result: dict) -> str:
    label = result["label"]
    emoji = LABEL_EMOJI.get(label, label)
    conf_pct = result["confidence"] * 100
    filled = int(conf_pct // 5)
    bar = "=" * filled + "-" * (20 - filled)
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
    if "raw_generation" in result:
        lines.append(f"  SLM Output  : '{result['raw_generation']}'")
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
        description="Interactive Marathi sentiment predictor for BERT and Sarvam-1 SLM."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Saved model directory or LoRA adapter (default: {DEFAULT_MODEL_DIR}).",
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
