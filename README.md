# Fine-Tuning, Step by Step 🎯

[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-2563eb?style=for-the-badge&logo=github)](https://rajeevranjanpandey.github.io/learn-fine-tuning/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(Pure%20Web)-blue.svg?style=for-the-badge)]()
[![PyTorch Compatible](https://img.shields.io/badge/Python-3.9%2B%20%7C%20PyTorch-ee4c2c?style=for-the-badge&logo=pytorch)](train.py)

> **An interactive, visual, and mathematical guide to fine-tuning modern language models.** Follow every tensor, gradient, optimizer step, and quantized weight on a real working model small enough to calculate by hand, verified down to two decimal places against PyTorch.

---

## 🌐 Live Interactive Website

Explore the full interactive lesson directly in your browser:
**👉 [https://rajeevranjanpandey.github.io/learn-fine-tuning/](https://rajeevranjanpandey.github.io/learn-fine-tuning/)**

---

## 📖 What is This Project?

Pre-trained Large Language Models (LLMs) know a vast amount of general language, but raw base models often simply repeat prompts or ramble without executing specific tasks. **Fine-Tuning** adapts pre-trained weights to specialize in a targeted skill, format, or tone using a modest, high-quality dataset.

While most tutorials treat fine-tuning as an opaque `trainer.train()` black box, **Fine-Tuning, Step by Step** demystifies the entire mechanics:
1. **Interactive Visualizations**: Watch forward passes, attention weights, gradient backpropagation, and weight updates animated step by step.
2. **Exact Hand-Calculated Arithmetic**: Every loss value, AdamW moment, LoRA low-rank product, and NF4 quantization level is traceable to exact formulas.
3. **Dual Execution Engine**:
   - **In-Browser Web Autodiff Engine**: Train LoRA, QLoRA, or full weights live in your browser using pure JavaScript.
   - **Reproducible Python Training Script (`train.py`)**: A self-contained script that trains the base Transformer and all fine-tuning variations on CPU in under 90 seconds.

---

## 👥 Who Can Use This?

| Audience | How to Use This Resource | Key Takeaway |
| :--- | :--- | :--- |
| **Students & Beginners** | Follow the **Core 13-Step Path** in Beginner level. Use the interactive "Play the picture" controls and quick-check quizzes. | Understand *why* fine-tuning works without getting overwhelmed by jargon. |
| **Machine Learning Engineers** | Use the **GPU Memory Calculator**, the **Troubleshooting Table**, and the **Production Deployment Checklist**. | Accurately budget GPU VRAM ($16\Psi$ rule) and debug loss spikes or catastrophic forgetting. |
| **Educators & Professors** | Present the interactive diagrams in lectures, assign the built-in practice math boxes, and use the 50-minute and 90-minute course lesson plans. | Give students an intuitive and mathematically grounded mental model of SFT, LoRA, and DPO. |
| **Researchers & Practitioners** | Explore the **Research Roadmap**, inspect the **Ablation Lab**, and test the **Mini-Projects** using `train.py`. | Benchmark parameter rank vs. forgetting, and evaluate low-bit quantization tradeoffs. |

---

## 🗺️ What is Covered? (Full Curriculum)

### Part 1: Why Fine-Tune?
- **Step 0: The Job** — Transforming text completion into task instruction execution.
- **Step 1: Pre-training vs. Fine-Tuning** — Leveraging transfer learning from trillions of tokens to thousands of examples.
- **Step 2: Prompting, RAG, or Fine-Tuning?** — Decision framework: when to engineer prompts, when to retrieve documents, and when to adapt weights.
- **Step 3: The Five Fine-Tuning Lanes** — Continued pre-training, Supervised Fine-Tuning (SFT), Preference Tuning (DPO/RLHF), Task Heads, and Knowledge Distillation.

### Part 2: Data Engineering & Preparation
- **Step 4: Defining Task & Success Metrics** — Setting exact match, token F1, and evaluation rubrics *before* collecting data.
- **Step 5: Collecting & Formatting JSONL** — Structuring conversational turns into clean JSON Lines records.
- **Step 6: Data Quality Funnel** — Deduplication, PII scrubbing, toxicity filtering, format validation, and test decontamination.
- **Step 7: Tokenization & Chat Templates** — Special tokens (`<user>`, `<asst>`, `</s>`) and the critical danger of template mismatch.
- **Step 8: Loss Masking** — Why training only on answer tokens is crucial (and proof of what happens when you don't).
- **Step 9: Sequence Length, Padding & Packing** — Block-diagonal attention masks and eliminating wasted pad tokens.
- **Step 10: Train, Validation, and Test Splits** — Preventing test-set contamination and honest leave-one-out validation.

### Part 3: The Training Loop & Optimization
- **Step 11: Forward Pass & Masked Cross-Entropy Loss** — From token embeddings through 2 Transformer blocks to $- \log p(y)$.
- **Step 41: Autoregressive Decoding & Temperature** — Greedy choice vs. multinomial sampling and temperature sharpening.
- **Step 12: Backpropagation & The Chain Rule** — Propagating error gradients back through frozen and trainable layers.
- **Step 13: Gradient Descent Types & The Gradient Lab** — Batch, mini-batch, and stochastic gradient descent on live loss surfaces.
- **Step 14: Batch Sizes & Gradient Accumulation** — Micro-batches, accumulation steps, and effective batch size calculation ($B_{\text{eff}} = B_{\text{micro}} \times k_{\text{accum}} \times N_{\text{GPU}}$).
- **Step 15: Optimizers (SGD, Momentum, Adam, AdamW)** — First and second moments ($m, v$), bias correction, and decoupled weight decay ($\lambda$).
- **Step 16: Learning Rate Schedules** — Warm-up stages and cosine decay curves.
- **Step 17: Gradient Clipping** — Rescaling exploding gradients while preserving vector direction.
- **Step 18: Mixed Precision (FP32, FP16, BF16)** — Exponent vs. mantissa trade-offs and FP32 master weight copies.
- **Step 19: Overfitting & Regularization** — Early stopping, dropout, and weight decay dynamics.
- **Step 20: Catastrophic Forgetting** — Measuring prior skill degradation and how PEFT mitigates it.

### Part 4: Parameter-Efficient Fine-Tuning (PEFT)
- **Step 21: GPU Memory Breakdown** — The 16-bytes-per-parameter rule ($2\text{B weights} + 2\text{B grads} + 12\text{B optimizer}$) + activation memory.
- **Step 22: Freezing Layers & Linear Probing** — Unfreezing heads vs. full fine-tuning.
- **Step 23: LoRA: Low-Rank Adaptation** — Decomposing $\Delta W = \frac{\alpha}{r} B \cdot A$ with $B$ initialized to zero.
- **Step 24: LoRA Hyperparameters** — Selecting rank $r$, scaling factor $\alpha$, dropout, and target projection matrices ($W_q, W_k, W_v, W_o, \text{FFN}$).
- **Step 25: Merging & Serving Adapters** — Weight merging for zero latency overhead and multi-tenant adapter swapping.
- **Step 26: Alternative PEFT Methods** — Prompt tuning, Prefix tuning, Bottleneck Adapters, $\text{IA}^3$, and DoRA (Weight-Decomposed LoRA).
- **Step 27: Gradient Checkpointing** — Trading compute for memory by storing $O(\sqrt{L})$ activations.

### Part 5: Quantization & QLoRA
- **Step 28: Fundamentals of Quantization** — Moving from 32-bit floats to 16, 8, and 4 bits.
- **Step 29: Quantization Mathematics** — Scale factors, zero points, rounding error, and symmetric vs. asymmetric mapping.
- **Step 30: Outliers & NormalFloat4 (NF4)** — Information-theoretic quantization for normally distributed weights vs. `LLM.int8()`.
- **Step 31: QLoRA** — Fine-tuning LoRA adapters over a frozen, 4-bit NF4 quantized base model.
- **Step 32: Deployment Quantization (GPTQ, AWQ, GGUF)** — Post-training quantization (PTQ) and quantization-aware training (QAT).

### Part 6: Distributed Scaling & Throughput
- **Step 33: Distributed Training (ZeRO & FSDP)** — ZeRO Stages 1, 2, and 3: sharding optimizer states, gradients, and model parameters.
- **Step 34: Throughput Optimization & MFU** — FlashAttention, sequence packing, and measuring non-padding tokens per second.

### Part 7: Preference Tuning & Alignment
- **Step 35: Reward Modeling** — Bradley-Terry preference probability models and the sigmoid loss $\mathcal{L}_{\text{RM}} = -\log \sigma(r_w - r_l)$.
- **Step 36: RLHF with PPO & The KL Leash** — Policy optimization, reward hacking, and the reference model penalty.
- **Step 37: Direct Preference Optimization (DPO)** — Eliminating the reward model and RL loop via closed-form policy optimization (plus ORPO and KTO).

### Part 8: Evaluation, Production & Ethics
- **Step 38: Rigorous Evaluation** — Exact match, token perplexity, regression test suites, and mitigating LLM-as-a-judge biases.
- **Step 39: Experiment Tracking** — Multi-seed validation, reporting means $\pm$ standard deviations, and error bars.
- **Step 40: Production Deployment & Model Cards** — KV cache sizing, responsible release checklists, and safety auditing.

---

## 🚀 Getting Started Locally

### 1. Run the Web Application
No build tools, bundlers, or servers required!
```bash
git clone https://github.com/rajeevranjanpandey/learn-fine-tuning.git
cd learn-fine-tuning

# Open directly in any modern browser:
open index.html        # On macOS
xdg-open index.html    # On Linux
start index.html       # On Windows
```
Or serve with python:
```bash
python3 -m http.server 8000
# Open http://localhost:8000 in your browser
```

### 2. Run the Exact Training Script (`train.py`)
Validate every single measurement, curve, and weight value:
```bash
# Requires only standard PyTorch (CPU is fast!)
pip install torch

python train.py
```
This produces `model.json` containing the pre-training curves, base weights, full fine-tuning, LoRA ($r=1$), and QLoRA weights, alongside all ablations.

---

## 🛠️ Student & Engineer Toolkit Included

- **Interactive Playground**: Slide the adapter scale from $0.00$ to $2.00$ and see next-word probabilities shift live.
- **In-Browser Training Lab**: Train the toy model live inside your browser using JavaScript autodiff.
- **VRAM Calculator**: Estimate total GPU memory across 1B, 7B, 13B, and 70B models for Full, LoRA, and QLoRA.
- **Gradient Descent 3D Contour Lab**: Step through batch, mini-batch, and SGD paths across real loss landscapes.
- **Final Comprehensive Quiz**: 10 mixed-topic assessment questions with instant feedback.
- **Interactive Glossary & Color Key**: 60+ terms linked directly to the steps where they appear.

---

## 📚 Landmark Papers & Citations

This educational project is built on peer-reviewed literature:
- **ULMFiT**: Howard & Ruder (ACL 2018) — *Universal Language Model Fine-tuning for Text Classification*
- **Adapter Layers**: Houlsby et al. (ICML 2019) — *Parameter-Efficient Transfer Learning for NLP*
- **LoRA**: Hu et al. (ICLR 2022) — *Low-Rank Adaptation of Large Language Models*
- **QLoRA**: Dettmers, Pagnoni, Holtzman, Zettlemoyer (NeurIPS 2023) — *Efficient Finetuning of Quantized LLMs*
- **DPO**: Rafailov et al. (NeurIPS 2023) — *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*
- **ZeRO**: Rajbhandari et al. (SC 2020) — *Memory Optimizations Toward Training Trillion Parameter Models*
- **FlashAttention**: Dao et al. (NeurIPS 2022) — *Fast and Memory-Efficient Exact Attention with IO-Awareness*

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE). Free for educational and commercial use.

---

**Created with precision by [Rajeev Ranjan Pandey](https://github.com/rajeevranjanpandey)**
