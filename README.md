# Transformer GPT from Scratch

A from-scratch implementation of a GPT-style decoder-only Transformer using Python, NumPy, and PyTorch. This repository explores the fundamentals of machine learning and deep learning, progressing from gradient descent and backpropagation to attention mechanisms, Transformer architecture, and autoregressive text generation.

The project is organized into modular, self-contained implementations covering ML fundamentals, data preprocessing, tokenization, Transformer building blocks, training, and inference. It also explores modern LLM techniques such as Grouped-Query Attention (GQA), RMSNorm, and Key-Value (KV) caching.

## Project Structure

```text
transformer-gpt-scratch/
├── foundations/     # Machine learning fundamentals
├── data/           # Tokenization and dataset preparation
├── model/          # Transformer and GPT building blocks
├── train.py        # Model training
├── generate.py      # Autoregressive text generation
└── requirements.txt # Project dependencies
```

## Features

- Implemented core machine learning concepts from scratch using NumPy
- Built a GPT-style decoder-only Transformer in PyTorch
- Implemented character-level tokenization and BPE-style tokenization experiments
- Developed causal self-attention, multi-head attention, and Grouped-Query Attention (GQA)
- Implemented positional encodings, LayerNorm, and RMSNorm
- Added KV caching for autoregressive inference
- Built a next-token prediction training pipeline using Cross-Entropy loss and AdamW
- Implemented autoregressive text generation with sampling
- Explored training diagnostics, activation statistics, and dead-ReLU detection

## Module Breakdown

### 1. `foundations/` — Machine Learning Fundamentals

| Module | Description |
|---|---|
| `gradient_descent.py` | Minimizes \(f(x)=x^2\) using gradient descent |
| `linear_regression.py`, `linear_regression_training.py`, `training_loop.py` | Linear regression, prediction, loss computation, derivatives, and training |
| `activations.py`, `softmax.py`, `neuron.py` | Activation functions, softmax, and single-neuron forward pass |
| `loss.py` | Binary cross-entropy loss implementation |
| `backprop.py`, `multi_layer_backprop.py` | Manual backpropagation for single- and multi-layer networks |
| `mlp.py` | Multi-layer perceptron implemented using NumPy |
| `weight_init.py` | Xavier/Glorot and related weight initialization methods |
| `pytorch_basics.py` | Core tensor operations and manipulation |
| `digit_classifier.py` | Digit classification model using `nn.Sequential` |
| `sentiment.py` | Embedding-based sentiment classification model |
| `training_diagnostics.py`, `dead_relu_detector.py` | Activation monitoring, training diagnostics, and dead-ReLU detection |

### 2. `data/` — Tokenization and Data Preparation

| Module | Description |
|---|---|
| `vocab.py` | Character-level vocabulary with `stoi` and `itos` mappings |
| `tokenizer.py` | BPE-style token merge learning |
| `tokenizer_utils.py` | Greedy tokenization using a fixed vocabulary |
| `dataset.py`, `loader.py` | Batch sampling for next-token prediction using input-target pairs |
| `nlp_preprocessing.py` | Preprocessing and encoding raw text into token IDs |

### 3. `model/` — Transformer Building Blocks

| Module | Description |
|---|---|
| `embeddings.py` | Token embedding lookup |
| `positional_encoding.py` | Sinusoidal positional encoding |
| `attention.py` | Single-head causal self-attention with Q/K/V projections, scaling, and masking |
| `multi_head_attention.py` | Multi-head self-attention with output projection |
| `grouped_query_attention.py` | Grouped-Query Attention (GQA) |
| `kv_cache.py` | Key-Value caching for autoregressive inference |
| `normalization.py`, `batch_normalization.py`, `rms_normalization.py` | LayerNorm, BatchNorm, and RMSNorm implementations |
| `transformer.py` | Pre-LayerNorm Transformer block with attention, feed-forward layers, and residual connections |
| `gpt.py` | Full GPT model with token embeddings, positional information, stacked Transformer blocks, final normalization, and vocabulary projection |

### 4. Training — `train.py`

The training pipeline implements next-token prediction using randomly sampled context windows from the training data.

- Samples input-target sequences for next-token prediction
- Computes logits and cross-entropy loss
- Optimizes model parameters using AdamW
- Tracks training loss across iterations

### 5. Inference — `generate.py`

The inference pipeline generates text autoregressively by predicting one token at a time.

- Generates tokens conditioned on a starting context
- Restricts input to the model's context window
- Samples the next token from the model's output probability distribution
- Supports KV caching to avoid recomputing previous key/value states during generation

## Architecture Overview

The model follows the decoder-only Transformer architecture used in GPT-style language models.

```text
Input Token IDs
      │
      ▼
Token Embeddings + Positional Encodings
      │
      ▼
┌──────────────────────────────┐
│   Transformer Block          │
│                             │
│   Layer Normalization        │
│   Causal Self-Attention     │
│   Residual Connection       │
│                             │
│   Layer Normalization        │
│   Feed-Forward Network      │
│   Residual Connection       │
└──────────────────────────────┘
      │
      ▼
  Repeat N Blocks
      │
      ▼
  Final Layer Normalization
      │
      ▼
  Linear Projection to Vocabulary
      │
      ▼
  Next-Token Logits
```

### Core Components

- **Token embeddings:** Convert token IDs into dense vector representations.
- **Positional encodings:** Incorporate sequence position information.
- **Causal self-attention:** Allow each token to attend only to previous tokens and itself.
- **Multi-head attention:** Capture different relationships between tokens through parallel attention heads.
- **Grouped-Query Attention:** Share key/value heads across multiple query heads to reduce KV-cache memory requirements.
- **Feed-forward network:** Apply nonlinear transformations independently at each sequence position.
- **Residual connections and normalization:** Stabilize optimization and improve information flow through the network.
- **KV caching:** Reuse previously computed key/value states during autoregressive generation.

### Training Objective

The model is trained using autoregressive next-token prediction. Given a sequence of tokens, it predicts the next token at every position.

The training objective is the cross-entropy loss:

\[
\mathcal{L}=-\frac{1}{T}\sum_{t=1}^{T}\log P(x_{t+1}\mid x_1,\ldots,x_t)
\]

The model learns to predict the next token based on the preceding context, with the objective optimized using AdamW.

## Getting Started

### Requirements

- Python 3.9+
- PyTorch
- NumPy
- Dependencies listed in `requirements.txt`

### Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/transformer-gpt-scratch.git
cd transformer-gpt-scratch
```

Create and activate a virtual environment:

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The model, training, and generation modules can be imported and used programmatically.

### Model Initialization

```python
from model.gpt import GPT

model = GPT(
    vocab_size=104,
    context_length=128,
    model_dim=252,
    num_blocks=6,
    num_heads=6,
)
```

### Training

```python
from train import Solution as Trainer

trainer = Trainer()

final_loss = trainer.train(
    model,
    data,
    epochs=1000,
    context_length=128,
    batch_size=32,
    lr=3e-4,
)
```

### Text Generation

```python
from generate import Solution as Generator

generator = Generator()

text = generator.generate(
    model,
    new_chars=500,
    context=context,
    context_length=128,
    int_to_char=itos,
)

print(text)
```

> **Note:** The example above assumes the dataset, encoded context, and vocabulary mappings (`data`, `context`, and `itos`) have already been prepared. Adjust the model configuration and function arguments to match the implementations in your repository.

## Learning Path

The repository is organized as a progressive learning journey:

1. **ML Fundamentals** — Gradient descent, linear regression, activation functions, loss functions, backpropagation, and MLPs
2. **Data & Tokenization** — Vocabulary construction, tokenization, text preprocessing, and batch generation
3. **Transformer Components** — Embeddings, positional encodings, causal attention, multi-head attention, and Transformer blocks
4. **GPT Architecture** — Combining Transformer blocks into a decoder-only language model
5. **Training & Inference** — Next-token prediction, optimization, autoregressive sampling, and text generation
6. **Modern LLM Techniques** — Grouped-Query Attention, RMSNorm, and KV caching

## Reproducibility

Random seeds are set in selected modules to support reproducible experiments. Results may still vary across hardware, PyTorch versions, and backend configurations.

## Future Improvements

- Add unit tests for model components and training behavior
- Add benchmark scripts for measuring training throughput and inference latency
- Compare inference performance with and without KV caching
- Evaluate memory and throughput improvements from mixed precision
- Add configurable model and training parameters through a CLI
- Add experiment tracking and checkpoint management
