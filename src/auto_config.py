"""
auto_config.py
==============
Hardware-adaptive training configuration and memory-validation dry run module.

Calculates optimal parameters (quantization, precision, batch size, gradient accumulation,
sequence length, gradient checkpointing, LoRA parameters, optimizer) based on detected
VRAM and compute capability, with explicit support for user overrides and memory dry runs.
"""

import gc
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))
from gpu_detector import HardwareProfile, detect_hardware, print_hardware_report

MODEL_CACHE_DIR: str = str(Path(__file__).parent.parent / "model")
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    # Hardware & Precision
    device: str  # "cuda" or "cpu"
    quantization: str  # "4bit_nf4" or "none"
    compute_dtype_str: str  # "bfloat16", "float16", "float32"
    torch_dtype: Any  # torch.bfloat16, torch.float16, torch.float32

    # Training Batch & Accumulation
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    max_seq_length: int
    gradient_checkpointing: bool

    # LoRA Adapter settings
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list

    # Optimizer & Scheduler
    optimizer: str
    learning_rate: float
    lr_scheduler_type: str
    warmup_steps: int

    # Checkpointing & Evaluation
    epochs: int
    eval_strategy: str
    save_strategy: str
    metric_for_best_model: str
    greater_is_better: bool
    save_total_limit: Optional[int]
    dataloader_num_workers: int

    # Metadata & Tracking
    user_overrides: Dict[str, Any]
    dry_run_passed: bool = False
    peak_vram_gb: float = 0.0

    def print_config(self) -> None:
        """Display clean, user-friendly configuration report."""
        print("========================================")
        print(" AUTOMATIC TRAINING CONFIGURATION")
        print("========================================")
        if self.user_overrides:
            print("  User Overrides Detected:")
            for k, v in self.user_overrides.items():
                print(f"    - {k} = {v}")
            print("----------------------------------------")
        print(f"  Training Device  : {self.device.upper()}")
        print(f"  Quantization     : {self.quantization}")
        print(f"  Compute Precision: {self.compute_dtype_str}")
        print(f"  Per-Device Batch : {self.batch_size}")
        print(f"  Grad Accum Steps : {self.gradient_accumulation_steps}")
        print(f"  Effective Batch  : {self.effective_batch_size}")
        print(f"  Max Seq Length   : {self.max_seq_length}")
        print(f"  Grad Checkpoint  : {'Enabled' if self.gradient_checkpointing else 'Disabled'}")
        print(f"  LoRA Rank (r)    : {self.lora_r}")
        print(f"  LoRA Alpha       : {self.lora_alpha}")
        print(f"  LoRA Dropout     : {self.lora_dropout}")
        print(f"  Optimizer        : {self.optimizer}")
        print(f"  Learning Rate    : {self.learning_rate}")
        print(f"  Num Epochs       : {self.epochs}")
        print(f"  Checkpoints      : Save every epoch")
        print("========================================\n")


def get_auto_config(
    profile: Optional[HardwareProfile] = None,
    epochs: int = 4,
    user_batch_size: Optional[int] = None,
    user_grad_accum: Optional[int] = None,
    user_seq_length: Optional[int] = None,
    user_lr: Optional[float] = None,
    user_lora_r: Optional[int] = None,
    user_grad_checkpointing: Optional[bool] = None,
    force_cpu: bool = False,
) -> TrainingConfig:
    """
    Generate hardware-adaptive training configuration.

    Parameters
    ----------
    profile : HardwareProfile | None
        Hardware capability details. If None, auto-detected.
    epochs : int
        Number of training epochs.
    user_batch_size, user_grad_accum, user_seq_length, user_lr, user_lora_r, user_grad_checkpointing : Optional
        Explicit user overrides.
    force_cpu : bool
        If True, forces CPU configuration.
    """
    if profile is None:
        profile = detect_hardware()

    user_overrides: Dict[str, Any] = {}

    # CPU or GPU determination
    use_cuda = profile.gpu_available and not force_cpu
    device = "cuda" if use_cuda else "cpu"

    if force_cpu:
        user_overrides["device"] = "cpu"

    # 1. Precision & Quantization
    if use_cuda:
        quantization = "4bit_nf4" if profile.bitsandbytes_available else "none"
        if profile.bf16_supported:
            compute_dtype_str = "bfloat16"
            torch_dtype = torch.bfloat16
        elif profile.fp16_supported:
            compute_dtype_str = "float16"
            torch_dtype = torch.float16
        else:
            compute_dtype_str = "float32"
            torch_dtype = torch.float32
        optimizer = "paged_adamw_8bit" if profile.bitsandbytes_available else "adamw_torch"
    else:
        quantization = "none"
        compute_dtype_str = "float32"
        torch_dtype = torch.float32
        optimizer = "adamw_torch"

    # 2. VRAM Tiering & Auto Heuristics
    vram = profile.free_vram_gb if profile.gpu_available else 0.0

    if not use_cuda or vram <= 7.0:  # e.g., RTX 3050 (6GB)
        auto_batch_size = 1
        auto_grad_accum = 16
        auto_seq_length = 512
        auto_grad_checkpointing = True
        auto_lora_r = 8
    elif vram <= 13.0:  # e.g., RTX 3060/3070/4060 (8GB - 12GB)
        auto_batch_size = 2
        auto_grad_accum = 8
        auto_seq_length = 1024
        auto_grad_checkpointing = True
        auto_lora_r = 16
    elif vram <= 20.0:  # e.g., RTX 3080/4070 (10GB - 16GB)
        auto_batch_size = 4
        auto_grad_accum = 4
        auto_seq_length = 1024
        auto_grad_checkpointing = True
        auto_lora_r = 16
    else:  # e.g., RTX 3090/4090/5090 (24GB - 32GB+)
        auto_batch_size = 8
        auto_grad_accum = 2
        auto_seq_length = 1024
        auto_grad_checkpointing = False
        auto_lora_r = 32

    # 3. Apply User Overrides
    if user_batch_size is not None:
        batch_size = user_batch_size
        user_overrides["batch_size"] = batch_size
    else:
        batch_size = auto_batch_size

    if user_grad_accum is not None:
        gradient_accumulation_steps = user_grad_accum
        user_overrides["gradient_accumulation_steps"] = gradient_accumulation_steps
    else:
        # Scale grad accumulation to maintain target effective batch size of ~16 if batch size was changed
        if user_batch_size is not None:
            gradient_accumulation_steps = max(1, 16 // batch_size)
        else:
            gradient_accumulation_steps = auto_grad_accum

    if user_seq_length is not None:
        max_seq_length = user_seq_length
        user_overrides["max_seq_length"] = max_seq_length
    else:
        max_seq_length = auto_seq_length

    if user_grad_checkpointing is not None:
        gradient_checkpointing = user_grad_checkpointing
        user_overrides["gradient_checkpointing"] = gradient_checkpointing
    else:
        gradient_checkpointing = auto_grad_checkpointing

    if user_lora_r is not None:
        lora_r = user_lora_r
        user_overrides["lora_r"] = lora_r
    else:
        lora_r = auto_lora_r

    if user_lr is not None:
        learning_rate = user_lr
        user_overrides["learning_rate"] = learning_rate
    else:
        learning_rate = 2e-4

    lora_alpha = lora_r * 2
    effective_batch_size = batch_size * gradient_accumulation_steps

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    return TrainingConfig(
        device=device,
        quantization=quantization,
        compute_dtype_str=compute_dtype_str,
        torch_dtype=torch_dtype,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        effective_batch_size=effective_batch_size,
        max_seq_length=max_seq_length,
        gradient_checkpointing=gradient_checkpointing,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        optimizer=optimizer,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=None,  # Keep all epoch checkpoints (checkpoint-epoch-1..4)
        dataloader_num_workers=0,
        user_overrides=user_overrides,
    )


def dry_run_memory_check(
    config: TrainingConfig,
    model_name: str = "sarvamai/sarvam-1",
) -> TrainingConfig:
    """
    Perform a dry-run memory validation test.

    Loads model & tokenizer, creates a dummy batch, runs forward/backward pass,
    measures peak VRAM, and adapts settings if CUDA OOM is detected.
    """
    if config.device != "cuda":
        log.info("CPU mode: skipping CUDA memory dry run.")
        config.dry_run_passed = True
        return config

    log.info("Starting memory validation dry run for '%s' …", model_name)
    max_attempts = 5
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        log.info(
            "Memory Test [Attempt %d/%d] — Batch Size: %d, Seq Len: %d, Grad Accum: %d, Grad Checkpoint: %s",
            attempt,
            max_attempts,
            config.batch_size,
            config.max_seq_length,
            config.gradient_accumulation_steps,
            config.gradient_checkpointing,
        )

        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()

            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=MODEL_CACHE_DIR)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load quantized model
            if config.quantization == "4bit_nf4":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=config.torch_dtype,
                    bnb_4bit_use_double_quant=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=config.torch_dtype,
                    trust_remote_code=True,
                    cache_dir=MODEL_CACHE_DIR,
                )
                model = prepare_model_for_kbit_training(model)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=config.torch_dtype,
                    trust_remote_code=True,
                    cache_dir=MODEL_CACHE_DIR,
                )

            if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

            lora_cfg = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)

            # Dummy forward & backward pass
            dummy_text = [
                "Classify the sentiment of the following Marathi text. Reply with ONLY ONE WORD from [Positive, Negative, Neutral].\n"
                "Text: हा चित्रपट अतिशय उत्कृष्ट आणि मनोरंजक आहे.\n"
                "Sentiment: Positive"
            ] * config.batch_size

            enc = tokenizer(
                dummy_text,
                return_tensors="pt",
                max_length=config.max_seq_length,
                padding="max_length",
                truncation=True,
            )

            input_ids = enc["input_ids"].to("cuda")
            attention_mask = enc["attention_mask"].to("cuda")
            labels = input_ids.clone()

            # Enable compute precision autocast
            autocast_dtype = config.torch_dtype if config.torch_dtype in (torch.bfloat16, torch.float16) else torch.float32
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            loss.backward()

            peak_bytes = torch.cuda.max_memory_allocated()
            peak_vram_gb = round(peak_bytes / (1024**3), 2)
            config.peak_vram_gb = peak_vram_gb

            log.info("Memory Test PASSED! Peak VRAM allocated: %.2f GB", peak_vram_gb)
            config.dry_run_passed = True

            # Cleanup
            del model, tokenizer, outputs, loss, input_ids, attention_mask, labels
            torch.cuda.empty_cache()
            gc.collect()

            return config

        except torch.cuda.OutOfMemoryError as e:
            log.warning("CUDA OOM detected during memory test!")
            torch.cuda.empty_cache()
            gc.collect()

            # Fallback ladder
            if config.batch_size > 1:
                old_bs = config.batch_size
                config.batch_size = max(1, config.batch_size // 2)
                config.gradient_accumulation_steps *= 2
                config.effective_batch_size = config.batch_size * config.gradient_accumulation_steps
                print(f"\nCUDA OOM detected! Automatically adapting...")
                print(f"Batch size: {old_bs} → {config.batch_size}")
                print(f"Gradient accumulation: {config.gradient_accumulation_steps // 2} → {config.gradient_accumulation_steps}\n")
            elif config.max_seq_length > 256:
                old_seq = config.max_seq_length
                config.max_seq_length = max(256, config.max_seq_length // 2)
                print(f"\nCUDA OOM detected! Automatically adapting...")
                print(f"Sequence length: {old_seq} → {config.max_seq_length}\n")
            elif not config.gradient_checkpointing:
                config.gradient_checkpointing = True
                print(f"\nCUDA OOM detected! Automatically adapting...")
                print(f"Gradient checkpointing: Disabled → Enabled\n")
            elif config.lora_r > 8:
                old_r = config.lora_r
                config.lora_r = 8
                config.lora_alpha = 16
                print(f"\nCUDA OOM detected! Automatically adapting...")
                print(f"LoRA rank: {old_r} → {config.lora_r}\n")
            else:
                log.error("Unable to reduce memory footprint further. Please check hardware.")
                raise e

        except Exception as e:
            log.error("Memory dry run error: %s", e)
            # Cleanup and pass through if non-OOM error (e.g., download issue)
            torch.cuda.empty_cache()
            gc.collect()
            config.dry_run_passed = False
            return config

    return config


if __name__ == "__main__":
    profile = print_hardware_report()
    config = get_auto_config(profile=profile, epochs=4)
    config.print_config()
