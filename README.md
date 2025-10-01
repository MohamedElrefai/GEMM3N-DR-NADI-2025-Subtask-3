# 🦄 GEMM3N-DR – Audio-Text Diacritic Restoration (NADI 2025)

This repository contains the official implementation of:

**📄 Paper Title:**
*Unicorn at NADI 2025 Subtask 3: GEMM3N-DR – Audio-Text Diacritic Restoration via Fine-tuning Multimodal Arabic LLM*

Our end-to-end system for **Arabic Diacritic Restoration** is organized into three distinct stages:

1. **Stage 1: Fine-tuning** – Train the Gemma-3N model with LoRA adapters on the official NADI 2025 dataset plus augmented audio/text.
2. **Stage 2: Inference & Evaluation** – Run inference on development/test sets with structured few-shot prompts.
3. **Stage 3: Results & Analysis** – Compute WER/CER, analyze hallucinations, and preserve code-switched (non-Arabic) words.

---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/MohamedElrefai/GEMM3N-DR-NADI-2025-Subtask-3 -b main
cd GEMM3N-DR
pip install -r requirments.txt
```

---

## 📂 Files Overview

### 🔹 Core Pipeline Scripts

* **`gemm3n_finetuning.py`**

  * Loads **Gemma-3N (instruction-tuned)** via `unsloth.FastModel`.
  * Applies **LoRA (rank=128, α=16)** adapters to both text and audio modules.
  * Integrates **CATT model outputs** for pseudo-diacritization of augmented text.
  * Formats NADI training data into **multimodal prompts** (audio + undiacritized text → diacritized text).
  * Saves best checkpoint at step **16500**:

    ```
    extract_lora128_complete_with_augmented_text_checkpoint_16500/
    ```

* **`inference_dev_model.py`**

  * Loads the **best checkpoint (16500)**.
  * Runs **few-shot inference** (7-shot structured examples).
  * Preserves **non-Arabic words** (English, French, etc.) in original form.
  * Produces results:

    * `results_temp_*.csv` → Reference vs Prediction vs Input.
    * `wer_temp_*.txt` → Final WER score (via `jiwer`).

### 🔹 Environment Setup

* **`requirments.txt`**

  * Contains all required dependencies:

    * **Model training:** `torch`, `transformers`, `unsloth`, `trl`, `peft`
    * **Dataset handling:** `datasets`, `protobuf`, `sentencepiece`
    * **Efficiency:** `xformers`, `bitsandbytes`, `hf_transfer`
    * **Evaluation:** `jiwer`

Install with:

```bash
pip install -r requirments.txt
```

---

## 🚀 Usage

### **Stage 1: Fine-tuning**

Run the training script:

```bash
python gemm3n_finetuning.py
```

This will:

* Fine-tune **Gemma-3N** with LoRA rank=128.
* Use **original + augmented datasets** (audio augmentations + CATT pseudo-diacritization).
* Save outputs to:

  * Checkpoints → `lora128_complete_with_augmented_text_checkpoint_16500/`
  * Final LoRA model → `extract_lora128_complete_with_augmented_text_checkpoint_16500/`

---

### **Stage 2: Inference & Evaluation**

Run inference on the NADI dev set:

```bash
python inference_dev_model.py
```

This will:

* Load the checkpoint-16500 model.
* Apply **structured prompts** with 7 examples.
* Evaluate performance across different temperatures (default = `0.001`).
* Save results to:

  * `results_temp_0.001_with_catt_inference.csv`
  * `wer_temp_0.001_with_catt_inference.txt`

---

### **Stage 3: Custom Inference (Optional)**

You can adapt the inference script to diacritize your **own Arabic text + audio**:

```python
from inference_dev_model import process_segment
output = process_segment("النص بدون تشكيل", audio_array, 0.001, duration_seconds=60)
print(output)
```

---

## 📊 Pipeline Summary

```text
Stage 1 → Fine-tuning (LoRA rank=128, augmented data)
Stage 2 → Inference (few-shot prompting, checkpoint-16500)
Stage 3 → Evaluation (WER, CER, error analysis)
```

---

## 🏆 Results

* **Baseline (Text-only, no fine-tuning):** WER = 71%, CER = 23%
* **Fine-tuned (LoRA rank=128 + 7-shot):** WER = 64%, CER = 15%
* Ranked **2nd place** in NADI 2025 Subtask 3.

---

## ✨ Key Features

* Multimodal **Audio + Text** input for diacritic restoration.
* **LoRA fine-tuning** for efficiency on Gemma-3N.
* **Data augmentation** (audio + pseudo-diacritized text).
* **Structured prompting** to reduce hallucination.
* Automatic **WER/CER computation** for evaluation.

