"""
train_sarvam.py
===============
Fine-tunes the generative Small Language Model (SLM) `sarvamai/sarvam-1` (2B)
on the Marathi Sentiment Dataset (MahaSent-MD) using 4-bit QLoRA (Parameter-Efficient Fine-Tuning).

Key Features:
- 4-bit NormalFloat (NF4) quantization via bitsandbytes
- bfloat16 compute dtype to prevent NaN loss spikes
- LoRA targeting all linear projection layers (q, k, v, o, gate, up, down)
- Instruction-style prompt formatting for causal sentiment classification
- SFTTrainer training loop with evaluation & checkpointing
- Generates and saves loss curve visualization in results/

Usage:
------
# Full fine-tuning (GPU with QLoRA)
python src/train_sarvam.py --data_dir data/

# Fast debug run (99 rows, 1 epoch)
python src/train_sarvam.py --data_dir data/ --debug
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

# Make src/ importable when called from project root
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import LABEL_NAMES, load_raw_dataset

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

try:
    from trl import SFTConfig, SFTTrainer
except ImportError:
    SFTConfig = None
    SFTTrainer = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = "sarvamai/sarvam-1"
DEFAULT_OUTPUT_DIR: str = "saved_models/sarvam-1-lora"
RESULTS_DIR: Path = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_TEMPLATE: str = (
    "Classify the sentiment of the following Marathi text. Reply with ONLY ONE WORD from [Positive, Negative, Neutral].\n"
    "Text: {text_sample}\n"
    "Sentiment: {label}"
)

# Mapping from class index (0=neg, 1=neu, 2=pos) or name to Capitalized prompt label
INT_TO_LABEL_STR: dict[int, str] = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}
STR_TO_LABEL_STR: dict[str, str] = {
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
    "-1": "Negative",
    "0": "Neutral",
    "1": "Positive",
}


# ---------------------------------------------------------------------------
# Loss Curve Plotting
# ---------------------------------------------------------------------------

def plot_loss_curves(log_history: list[dict], output_plot_path: Path, model_tag: str = "sarvam-1") -> None:
    """Save a training / validation loss curve PNG from Trainer log history."""
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("step", 0))
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", 0))
            eval_losses.append(entry["eval_loss"])

    if not train_losses:
        log.warning("No loss entries found in log history — skipping loss plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_steps, train_losses, label="Training Loss", color="#2196F3", linewidth=2)
    if eval_losses:
        ax.plot(
            eval_steps,
            eval_losses,
            label="Validation Loss",
            color="#FF5722",
            linewidth=2,
            linestyle="--",
            marker="o",
            markersize=5,
        )

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Training & Validation Loss — {model_tag} (QLoRA)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Loss curves saved → %s", output_plot_path)


# ---------------------------------------------------------------------------
# Dataset Formatting
# ---------------------------------------------------------------------------

def format_dataset_for_sft(raw_ds: DatasetDict) -> DatasetDict:
    """
    Format dataset splits into instruction text format.
    Format:
      "Classify the sentiment of the following Marathi text. Reply with ONLY ONE WORD from [Positive, Negative, Neutral].\nText: {text_sample}\nSentiment: {label}"
    """
    formatted_dict = {}

    for split_name, ds in raw_ds.items():
        formatted_texts = []
        for sample in ds:
            text_val = sample.get("text", "")
            raw_label = sample.get("label", "")

            if isinstance(raw_label, (int, torch.Tensor)):
                label_str = INT_TO_LABEL_STR.get(int(raw_label), "Neutral")
            elif isinstance(raw_label, str):
                label_str = STR_TO_LABEL_STR.get(raw_label.lower().strip(), raw_label.capitalize())
            else:
                label_str = "Neutral"

            prompt_text = PROMPT_TEMPLATE.format(text_sample=text_val, label=label_str)
            formatted_texts.append(prompt_text)

        formatted_dict[split_name] = Dataset.from_dict({"text": formatted_texts})
        log.info("Formatted %-12s split: %d samples", split_name, len(formatted_texts))

    return DatasetDict(formatted_dict)


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train_sarvam(
    model_name: str = DEFAULT_MODEL,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    data_dir: str = None,
    debug: bool = False,
    epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    force_cpu: bool = False,
) -> None:
    """Fine-tune Sarvam-1 using 4-bit QLoRA."""
    if LoraConfig is None or SFTTrainer is None:
        raise ImportError(
            "peft and trl are required for training Sarvam-1. Please run: pip install peft trl bitsandbytes"
        )

    use_cuda = torch.cuda.is_available() and not force_cpu
    log.info("Starting Sarvam-1 training pipeline (CUDA=%s) …", use_cuda)

    # 1. Dataset loading & formatting
    raw_ds = load_raw_dataset(debug=debug, data_dir=data_dir)
    formatted_ds = format_dataset_for_sft(raw_ds)
    train_ds = formatted_ds["train"]
    eval_key = "validation" if "validation" in formatted_ds else "test"
    eval_ds = formatted_ds[eval_key]

    log.info("Sample prompt for training:\n%s", train_ds[0]["text"])

    # 2. Tokenizer setup
    log.info("Loading tokenizer for '%s' …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Model loading with 4-bit Quantization
    log.info("Loading base causal model '%s' …", model_name)
    if use_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        log.warning("CUDA unavailable or CPU forced: loading in full precision on CPU (slow).")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    # Synchronize pad token id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    # 4. LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable_params, total_params = model.get_nb_trainable_parameters()
    log.info(
        "LoRA Parameters: %s trainable / %s total (%.2f%% trainable)",
        f"{trainable_params:,}",
        f"{total_params:,}",
        (trainable_params / total_params) * 100,
    )

    # 5. Training Arguments / SFTConfig
    num_epochs = 1 if debug else epochs
    config_cls = SFTConfig if SFTConfig is not None else TrainingArguments
    config_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": num_epochs,
        "lr_scheduler_type": "cosine",
        "bf16": use_cuda,
        "fp16": False,
        "optim": "paged_adamw_8bit" if use_cuda else "adamw_torch",
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 2,
        "logging_steps": 5 if debug else 50,
        "warmup_steps": 5 if debug else 100,
        "report_to": "none",
        "seed": 42,
        "data_seed": 42,
        "dataloader_num_workers": 0,
    }
    if SFTConfig is not None:
        config_kwargs["dataset_text_field"] = "text"
        config_kwargs["max_length"] = 256

    training_args = config_cls(**config_kwargs)

    # 6. SFTTrainer
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "args": training_args,
    }
    if SFTConfig is not None:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["dataset_text_field"] = "text"
        trainer_kwargs["max_seq_length"] = 256
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    # 7. Train
    log.info("Starting SFT training for %d epochs …", num_epochs)
    train_result = trainer.train()

    # 8. Save best LoRA adapter & tokenizer
    log.info("Saving best LoRA adapter to '%s' …", output_dir)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 9. Evaluate
    log.info("Running evaluation on '%s' split …", eval_key)
    eval_metrics = trainer.evaluate()
    log.info("Evaluation metrics: %s", eval_metrics)

    # 10. Save summary & loss curve
    summary = {**train_result.metrics, **eval_metrics}
    summary_csv = RESULTS_DIR / "sarvam-1_train_metrics.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    log.info("Saved training metrics → %s", summary_csv)

    loss_plot_path = RESULTS_DIR / "sarvam-1_loss_curves.png"
    plot_loss_curves(trainer.state.log_history, loss_plot_path, model_tag="sarvam-1")

    log.info("Sarvam-1 fine-tuning completed successfully!")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune sarvamai/sarvam-1 on Marathi sentiment with 4-bit QLoRA."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Base model (default: {DEFAULT_MODEL})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"LoRA save dir (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--data_dir", type=str, default=None, help="Local CSV data directory")
    parser.add_argument("--debug", action="store_true", help="Quick 99-row sanity check")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size (default: 2)")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    return parser.parse_args()


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
        learning_rate=args.lr,
        force_cpu=args.cpu,
    )
