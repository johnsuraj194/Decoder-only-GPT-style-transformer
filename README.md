# Transformer GPT from Scratch

A step-by-step implementation of a character-level GPT built from first principles in Python, NumPy, and PyTorch. The repository is organized as a progression of small, self-contained modules: it starts with machine learning fundamentals (gradient descent, backpropagation, MLPs), moves through data preparation and tokenization, builds up the core Transformer components (attention, normalization, positional encodings), and ends with a full GPT model plus training and text-generation loops.

## Project Structure

```
transformer-gpt-scratch/
├── foundations/     # ML fundamentals, from scratch
├── data/            # Tokenization and dataset preparation
├── model/           # Transformer / GPT building blocks
├── train.py         # Training loop for the GPT model
├── generate.py      # Autoregressive text generation
└── requirements.txt
```

### `foundations/` — Machine Learning Fundamentals

| Module | Description |
| --- | --- |
| `gradient_descent.py` | Minimizing f(x) = x² with vanilla gradient descent |
| `linear_regression.py`, `linear_regression_training.py`, `training_loop.py` | Linear regression: prediction, MSE, derivatives, and a full training loop |
| `activations.py`, `softmax.py`, `neuron.py` | Sigmoid, ReLU, softmax, and a single neuron forward pass |
| `loss.py` | Binary cross-entropy loss |
| `backprop.py`, `multi_layer_backprop.py` | Manual backpropagation for single- and multi-layer networks |
| `mlp.py` | Multi-layer perceptron forward pass in NumPy |
| `weight_init.py` | Xavier/Glorot and related weight initialization schemes |
| `pytorch_basics.py` | Core tensor operations (reshape, mean, concat, etc.) |
| `digit_classifier.py` | MNIST-style digit classifier with `nn.Sequential` |
| `sentiment.py` | Simple embedding-based sentiment model |
| `training_diagnostics.py`, `dead_relu_detector.py` | Activation statistics and dead-ReLU detection |

### `data/` — Tokenization and Data Preparation

| Module | Description |
| --- | --- |
| `vocab.py` | Character-level vocabulary (`stoi` / `itos` mappings) |
| `tokenizer.py` | Byte-pair-encoding style merge learning |
| `tokenizer_utils.py` | Greedy tokenization helpers using a fixed vocabulary |
| `dataset.py`, `loader.py` | Random batch sampling of (input, target) pairs for next-token prediction |
| `nlp_preprocessing.py` | Building an encoded dataset from raw sentences |

### `model/` — Transformer Building Blocks

| Module | Description |
| --- | --- |
| `embeddings.py` | Embedding table lookup |
| `positional_encoding.py` | Sinusoidal positional encodings |
| `attention.py` | Single-head causal self-attention (Q/K/V, scaling, masking) |
| `multi_head_attention.py` | Multi-headed self-attention with output projection |
| `grouped_query_attention.py` | Grouped-query attention (GQA), as used in modern LLMs |
| `kv_cache.py` | Key/value cache for efficient autoregressive inference |
| `normalization.py`, `batch_normalization.py`, `rms_normalization.py` | LayerNorm, BatchNorm, and RMSNorm implementations |
| `transformer.py` | Pre-LN Transformer block (attention + feed-forward with residual connections) |
| `gpt.py` | Full GPT: token + position embeddings, stacked Transformer blocks, final LayerNorm, and vocabulary projection |

### Top Level

- **`train.py`** — Training loop for the GPT model: samples random context windows, computes cross-entropy loss over next-token predictions, and optimizes with AdamW.
- **`generate.py`** — Autoregressive character generation: repeatedly feeds the last `context_length` tokens through the model, samples from the softmax distribution, and decodes the result back to text.

## Architecture Overview

The GPT model follows the decoder-only Transformer design:

1. **Token + positional embeddings** map input character IDs into `model_dim`-dimensional vectors.
2. **Transformer blocks** (Pre-LN) apply masked multi-head self-attention followed by a position-wise feed-forward network, each wrapped in a residual connection.
3. **Final LayerNorm + linear projection** produce logits over the vocabulary for every position.

Training uses the standard next-token-prediction objective: for each position in a context window, the model predicts the following character, and cross-entropy loss is averaged over all positions in the batch.

## Getting Started

### Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Usage

Each module exposes a `Solution` class (or an `nn.Module`) that can be imported and used directly. For example, building the GPT model and generating text:

```python
import torch
from model.gpt import GPT
from train import Solution as Trainer
from generate import Solution as Generator

model = GPT(vocab_size=104, context_length=128, model_dim=252, num_blocks=6, num_heads=6)

# Train on a 1D tensor of encoded character IDs
final_loss = Trainer().train(model, data, epochs=1000, context_length=128, batch_size=32, lr=3e-4)

# Generate new text from a starting context
text = Generator().generate(model, new_chars=500, context=context, context_length=128, int_to_char=itos)
print(text)
```

Note: many modules set `torch.manual_seed(0)` internally so their outputs are deterministic and reproducible.

## Learning Path

The modules are designed to be read in order of increasing complexity:

1. **Foundations** — gradient descent → linear regression → activations and losses → backpropagation → MLPs → PyTorch basics
2. **Data** — vocabulary construction → BPE tokenization → batch loading
3. **Model** — embeddings → positional encodings → single-head attention → multi-head attention → Transformer block → GPT
4. **Training & inference** — the training loop in `train.py`, then sampling in `generate.py`
5. **Modern extensions** — RMSNorm, grouped-query attention, and KV caching, which power today's production LLMs
