<h1 align="center">
  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="assets/banner/mlp-dark.svg"
    >
    <img
      alt="multilayer-perceptron"
      src="assets/banner/mlp-light.svg"
      width="500"
    >
  </picture>
  <br>
  <small>multilayer-perceptron: simple neural network</small>
</h1>

<p align=center>
    <picture>
      <img
        src="https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white"
        alt="Python"
      >
    </picture>
    <picture>
      <img
        src="https://img.shields.io/badge/Configuration-Pkl-74944E?logo=pkl&logoColor=white"
        alt="Pkl"
      >
    </picture>
    <picture>
      <img
        src="https://github.com/lucas-ht/multilayer-perceptron/actions/workflows/test.yaml/badge.svg"
        alt="Test Status"
      >
    </picture>
</p>


This project implements a simple **Multilayer Perceptron (MLP)**, a fundamental type of neural network commonly used in machine learning for classification tasks. Built from scratch using NumPy, it demonstrates core concepts like forward/backward propagation, gradient descent, and various activation functions—without relying on high-level frameworks like TensorFlow or PyTorch.


## Table of Contents

- [Features](#features)
- [Installation & Usage](#installation--usage)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Subject](#subject)
- [Acknowledgements](#acknowledgements)


## Features

- **Custom Neural Network**: Fully-connected layers with configurable architecture
- **Multiple Activation Functions**: ReLU, Sigmoid, Softmax
- **Optimizers**: Gradient Descent, Adam
- **Regularization**: L1 (Lasso) and L2 (Ridge) regularization
- **Training Features**: Mini-batch training, early stopping, gradient clipping
- **Configuration**: Model setup via `.pkl` configuration files
- **Visualization**: Automatic metric plotting (loss, accuracy, precision, recall)
- **Fast Setup**: Uses `uv` for lightning-fast dependency management


## Installation & Usage

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Quick Install

```bash
git clone https://github.com/lucas-ht/multilayer-perceptron.git
cd multilayer-perceptron
uv sync
```

For detailed configuration options and advanced usage, see the [complete documentation](/docs/usage.md).


## Quick Start

### 1. Split your dataset

```bash
uv run mlp.py split data/data.csv --test-size 0.2
```

### 2. Train the model

```bash
uv run mlp.py train data/train.csv model.test.pkl
```

### 3. Make predictions

```bash
uv run mlp.py predict data/test.csv models/model.json
```


## Architecture

The MLP consists of:
- **Input Layer**: Accepts feature vectors from the dataset
- **Hidden Layers**: Learns non-linear representations using activation functions
- **Output Layer**: Produces predictions (typically with softmax for classification)

Each layer performs:
1. **Linear transformation**: $` z = Wx + b `$
2. **Activation**: $` a = \sigma(z) `$
3. **Backpropagation**: Gradients computed via chain rule

### Training Process

1. **Forward propagation**: Pass inputs through layers to compute predictions
2. **Loss calculation**: Measure error using cross-entropy loss
3. **Backpropagation**: Compute gradients of loss w.r.t. weights
4. **Weight update**: Apply optimizer (GD or Adam) to minimize loss
5. **Regularization**: Optional L1/L2 penalties to prevent overfitting

Key hyperparameters (configurable in `.pkl` files):
- Learning rate
- Batch size
- Number of epochs
- Layer sizes and activation functions
- Regularization strength
- Early stopping patience


## Acknowledgements

This project is part of the 42 School curriculum. Special thanks to the 42 School community for their support and resources.
