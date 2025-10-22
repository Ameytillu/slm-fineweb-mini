# Small Language Model (SLM) – FineWeb 100K

A compact, end-to-end Small Language Model (SLM) trained on **100,000 samples** from the [HuggingFace FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb).  
This project demonstrates building a tokenizer, training a causal LM, and evaluating its performance using perplexity and loss metrics.

---

## Project Overview

This repository contains code and results for training a **custom tokenizer** and **language model** using a manageable subset of a large-scale dataset.  
The goal is to simulate the process of building a small-scale LLM from scratch.

### Key Stages
1. **Dataset Streaming** – Efficiently load 100K samples using streaming to avoid full download.  
2. **Sanity Checks** – Validate word count and token statistics before training.  
3. **Tokenizer Training** – Create a new tokenizer trained from scratch on the dataset.  
4. **Model Training** – Train a small Transformer-based causal language model using `Trainer`.  
5. **Evaluation** – Compute evaluation loss, perplexity (PPL), and token-level accuracy.

---

## Tech Stack

| Component | Library |
|------------|----------|
| Tokenization | `tokenizers`, `nltk`, `datasets` |
| Model Architecture | `transformers` (HuggingFace) |
| Training Framework | `Trainer` API |
| Dataset | `HuggingFaceFW/fineweb` (train split, 100K subset) |
| Environment | Google Colab / Jupyter Notebook |

---

## Evaluation Results

| Metric | Value | Description |
|:--|:--|:--|
| **Evaluation Loss** | `5.3089` | Average cross-entropy loss over validation set |
| **Perplexity (PPL)** | `202.12` | Measure of model uncertainty |
| **Token Accuracy (approx.)** | `≈ 35-55% (expected)` | Proportion of correct next-token predictions |

> A PPL of **~200** indicates moderate predictive ability — expected for a small model trained on 100K examples.  
> With more data or epochs, the model can achieve significantly better fluency and lower loss.

---

## Interpretation

- **Eval Loss (↓)** – Lower means the model predicts next tokens more confidently.  
- **Perplexity (↓)** – Indicates how “surprised” the model is by unseen text.  
- **Accuracy (↑)** – Measures next-token correctness (approximation only).

**Current results** reflect that the model has learned meaningful token-level dependencies but lacks deep contextual fluency due to limited data and epochs — perfect for educational and research demonstration.

---

## Future Improvements

- Increase training data from **100K → 500K+ samples**.  
- Train for **3–5 epochs** with gradual learning-rate decay.  
- Adopt **Byte-Pair Encoding (BPE)** or **SentencePiece** for tokenizer robustness.  
- Integrate **LoRA fine-tuning** on domain-specific corpora (e.g., hospitality, travel, customer reviews).  
- Add **Streamlit demo** to visualize model predictions interactively.



