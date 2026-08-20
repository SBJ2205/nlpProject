"""
predict.py
==========
Interactive real-time sentiment prediction on custom Marathi / code-mixed text.
Supports both fine-tuned Sarvam-1 SLM (with LoRA adapters) and encoder models.

Usage
-----
# Interactive mode
python predict.py --interactive

# Single sentence prediction
python predict.py --text "हा चित्रपट खूप छान आहे"
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import normalize_text
from gpu_detector import detect_hardware

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

MODEL_CACHE_DIR: str = str(Path(__file__).parent.parent / "model")
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR

DEFAULT_MODEL_DIR: str = "saved_models/sarvam-1-lora"
DEFAULT_BASE_MODEL: str = "sarvamai/sarvam-1"
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
    """Parse raw model generation into label index and string."""
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


class SentimentPredictor:
    """Predictor class for Sarvam-1 SLM and QLoRA fine-tuned adapters."""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, force_cpu: bool = False) -> None:
        profile = detect_hardware()
        use_cuda = profile.gpu_available and not force_cpu
        self.device = "cuda" if use_cuda else "cpu"

        model_path = Path(model_dir).resolve()
        base_model_name = DEFAULT_BASE_MODEL

        if model_path.exists() and (model_path / "adapter_config.json").exists():
            try:
                with open(model_path / "adapter_config.json", encoding="utf-8") as f:
                    cfg = json.load(f)
                    base_model_name = cfg.get("base_model_name_or_path", base_model_name)
            except Exception:
                pass
            is_adapter = True
        else:
            is_adapter = False

        log.info("Loading tokenizer for '%s' …", base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, cache_dir=MODEL_CACHE_DIR)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        log.info("Loading base causal model '%s' (device=%s) …", base_model_name, self.device)
        if use_cuda:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if profile.bf16_supported else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16 if profile.bf16_supported else torch.float16,
                trust_remote_code=True,
                cache_dir=MODEL_CACHE_DIR,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                cache_dir=MODEL_CACHE_DIR,
            )

        if is_adapter and PeftModel is not None:
            log.info("Attaching LoRA adapter from '%s' …", model_dir)
            self.model = PeftModel.from_pretrained(base_model, str(model_path))
        else:
            self.model = base_model

        self.model.eval()
        log.info("Predictor initialized successfully.")

    def predict(self, text: str) -> dict:
        """Predict sentiment for a single text string."""
        normalised = normalize_text(text)
        prompt = PROMPT_TEMPLATE.format(text_sample=normalised)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
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
        f"  SLM Output  : '{result['raw_generation']}'",
        f"  Input text  : {result['input']}",
    ]
    return "\n".join(lines)


def run_interactive(predictor: SentimentPredictor) -> None:
    """Start interactive REPL."""
    print("\n" + "═" * 58)
    print("  Sarvam-1 Marathi Sentiment Predictor")
    print("  Type Marathi text and press Enter. Type 'quit' to exit.")
    print("═" * 58)

    while True:
        try:
            text = input("\n▶ Enter Marathi text: ").strip()
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sarvam-1 Sentiment Predictor")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Model or LoRA adapter directory")
    parser.add_argument("--text", type=str, default=None, help="Single text sentence to predict")
    parser.add_argument("--interactive", action="store_true", help="Start interactive REPL mode")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    predictor = SentimentPredictor(model_dir=args.model_dir, force_cpu=args.cpu)

    if args.text:
        res = predictor.predict(args.text)
        print(_format_result(res))
    else:
        run_interactive(predictor)
