# Hardware-Adaptive Sarvam-1 QLoRA Fine-Tuning Framework

A production-grade, memory-aware fine-tuning framework for [`sarvamai/sarvam-1`](https://huggingface.co/sarvamai/sarvam-1) (2B Small Language Model) tailored for Marathi Sentiment Classification using the **MahaSent** dataset.

The core principle of this framework is:
> **The dataset remains fixed. The codebase automatically adapts itself to your hardware.**

---

## 🌟 Key Features

- 🤖 **Zero-Config Hardware Auto-Detection**: Automatically detects GPU vendor, total/free VRAM, CUDA compute capability, FP16/BF16 precision support, PyTorch, and `bitsandbytes` availability.
- ⚡ **Dynamic VRAM Adaptation**: Automatically computes optimal batch size, gradient accumulation, sequence length, LoRA rank, gradient checkpointing, and 8-bit paged optimizer without requiring manual tuning.
- 🛡️ **Memory Validation Dry-Run**: Performs a pre-flight forward/backward pass memory test on a representative mini-batch and automatically applies fallback strategies if CUDA OOM occurs.
- 🔒 **Read-Only Dataset Integrity**: Strictly leaves `MahaSent_All_Train.csv`, `MahaSent_All_Val.csv`, and `MahaSent_All_Test.csv` untouched on disk. All instruction prompt formatting occurs dynamically in-memory.
- 📊 **Baseline vs Fine-Tuned Comparison**: Evaluates both the zero-shot base Sarvam-1 model and your fine-tuned LoRA model on `MahaSent_All_Test.csv` side-by-side.
- 💬 **Interactive & CLI Sentiment Inference**: Real-time Marathi sentiment prediction via single text arguments or interactive REPL prompt.
- 📁 **Localized Model Storage**: Caches base model weights directly in `model/` inside the project folder.

---

## 💻 Supported Hardware

The framework dynamically scales across the entire spectrum of NVIDIA GPUs without modifying any code:

| Hardware Tier | GPU Examples | Auto Batch Size | Auto Grad Accum | Auto Seq Length | Grad Checkpoint | LoRA Rank ($r$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Entry Level (VRAM $\le$ 6 GB)** | RTX 3050 (6GB), GTX 1660 | 1 | 16 | 512 | Enabled | 8 |
| **Mid-Range (VRAM 8–12 GB)** | RTX 3060, RTX 3070, RTX 4060 | 2 | 8 | 1024 | Enabled | 16 |
| **High-End (VRAM 12–16 GB)** | RTX 3080, RTX 4070 | 4 | 4 | 1024 | Enabled | 16 |
| **Enthusiast (VRAM 24–32+ GB)** | RTX 3090, RTX 4090, RTX 5090 | 8 | 2 | 1024 | Disabled | 32 |

---

## 🖥️ Running on a New System (Zero-Config Setup)

If you transfer or clone this project to a new system (e.g. cloud GPU, workstation, or another computer), follow these 3 steps:

### Step 1: Environment & Dependency Setup

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# Git Bash / Linux:    source .venv/Scripts/activate  (or source .venv/bin/activate)

# 3. Install PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install framework dependencies
pip install -r requirements.txt
```

### Step 2: Dataset Verification

Ensure the 3 CSV files are located in `dataset/`:
```text
dataset/
├── MahaSent_All_Train.csv   (48,114 training samples)
├── MahaSent_All_Val.csv     (6,000 validation samples)
└── MahaSent_All_Test.csv    (6,750 test samples)
```

### Step 3: Run Training

```bash
python train_sarvam.py --epochs 4
```

> **Note for Git Bash users**: Use forward slashes when executing with `.venv`:
> `.venv/Scripts/python.exe train_sarvam.py --epochs 4`

---

## 🏋️ Training Options & Overrides

### Automatic Default
```bash
python train_sarvam.py --epochs 4
```

### Fast 1-Epoch Test Run (~6 hours on RTX 3050 6GB)
```bash
python train_sarvam.py --epochs 1
```

### Quick Sanity Debug Run (99 samples)
```bash
python train_sarvam.py --debug
```

### Manual Parameter Overrides (for advanced users)
```bash
python train_sarvam.py \
    --epochs 4 \
    --batch-size 2 \
    --seq-length 1024 \
    --lora-r 16
```

---

## 📊 Evaluation & Baseline Comparison

Evaluate the fine-tuned model and compare it against the original zero-shot base Sarvam-1 model on `MahaSent_All_Test.csv`:

```bash
python evaluate_sarvam.py
```

### Sample Comparison Output

```text
==================================================
 BASELINE VS FINE-TUNED MODEL COMPARISON
==================================================
Metric             Original        Fine-tuned     
--------------------------------------------------
Accuracy           0.3939          0.8142         
Macro-F1           0.2978          0.8085         
Precision          0.4137          0.8110         
Recall             0.3950          0.8065         
==================================================
```

Artifacts generated in `results/`:
- `results/sarvam-1_comparison.csv`: Metric comparison.
- `results/sarvam-1_confusion_matrix.png`: Dual raw & normalized confusion matrix heatmaps.
- `results/sarvam-1_per_class_metrics.csv`: Detailed per-class metrics.

---

## 🔮 Sentiment Prediction / Inference

### Single Sentence Prediction
```bash
python predict.py --text "हा चित्रपट खूप छान आणि मनोरंजक आहे"
```

### Interactive REPL Mode
```bash
python predict.py --interactive
```

---

## 🧠 QLoRA & VRAM Adaptation Mechanics

1. **4-bit NormalFloat (NF4) Quantization**: Compresses base 2B model weights from 16-bit to 4-bit, saving ~75% model VRAM.
2. **Double Quantization & 8-bit Paged Optimizer**: Offloads optimizer state spikes cleanly.
3. **Pre-flight Dry Run**: Runs a test forward/backward pass and automatically reduces batch size or sequence length if CUDA OOM is encountered.

---

## 📄 License & Attribution

- Base Model: [`sarvamai/sarvam-1`](https://huggingface.co/sarvamai/sarvam-1) by Sarvam AI.
- Dataset: L3Cube MahaSent-MD dataset.
