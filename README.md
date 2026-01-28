# LMD4MLTraining

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://LJS42.github.io/LMD4MLTraining.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://LJS42.github.io/LMD4MLTraining.jl/dev/)
[![Build Status](https://github.com/LJS42/LMD4MLTraining.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/LJS42/LMD4MLTraining.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/LJS42/LMD4MLTraining.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/LJS42/LMD4MLTraining.jl)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Live-Monitoring-Debugging-of-ML-Training
LMD4MLTraining.jl is a Julia package for live monitoring and visual debugging of neural network training in Flux.jl.

The package is inspired by the Python package cockpit and aims to provide insight into training dynamics by visualizing diagnostic quantities while training is running.

## Motivation
When training neural networks, issues such as unstable optimization, exploding gradients, or stalled learning are often only detected after training has finished. This package addresses this problem by providing live, interactive visualizations of important training metrics.

## Installation

To install the package, run the following in the Julia REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/LJS42/LMD4MLTraining.jl")
```

## Features
Currently implemented features include:

- Integration with standard Flux/Zygote training loops
- Live visualization using WGLMakie.jl
- Monitoring of user defined quantities: loss, gradient norm, distance, update size, norm test and gradient history.
- Modular design for adding additional quantities and visual instruments

## Getting Started

### Core Concepts

The central object in `LMD4MLTraining` is the `Learner`. It bundles everything needed for training and monitoring:

- **Model**: The Flux model to be trained.
- **Data Loader**: An iterable (like `Flux.DataLoader`) providing training batches.
- **Loss Function**: A function `f(ŷ, y)` that returns a vector of losses for the batch.
- **Optimizer**: The optimizer state (from `Flux.setup`).
- **Quantities**: A list of metrics to monitor during training.

### Training

To start training with live monitoring, use the `train!` function:

```julia
train!(learner, epochs, with_plots)
```

- `learner`: Your `Learner` instance.
- `epochs`: Number of training epochs.
- `with_plots`: Boolean. If `true`, starts a WGLMakie dashboard in your browser (or VS Code plot pane).

### Available Quantities

The package provides several diagnostic quantities inspired by the "Cockpit" paper:

| Quantity | Description |
| :--- | :--- |
| `LossQuantity()` | Tracks the training loss over time. |
| `GradNormQuantity()` | Monitors the L2 norm of the gradients. |
| `DistanceQuantity()` | Measures the L2 distance from the initialization. |
| `UpdateSizeQuantity()` | Tracks the L2 norm of the parameter updates. |
| `NormTestQuantity()` | Computes the "norm test" (checks if the gradient is dominated by noise). |
| `GradHist1dQuantity()` | Visualizes the 1D distribution of gradient elements. |

## Quick Start

You can run the provided MNIST example to see the dashboard in action without even cloning the repository. Copy and paste the following command into your terminal:

```bash
julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.add([Pkg.PackageSpec(url="https://github.com/LJS42/LMD4MLTraining.jl"), Pkg.PackageSpec(name="Flux"), Pkg.PackageSpec(name="MLDatasets")]); include(download("https://raw.githubusercontent.com/LJS42/LMD4MLTraining.jl/main/examples/mnist.jl"))'
```

If you have already cloned the repository, you can run it using:

```bash
# run from the repository root
julia --project=examples -e 'import Pkg; Pkg.instantiate(); include("examples/mnist.jl")'
```

Alternatively, include it in your own training loop:

```julia
using LMD4MLTraining
using Flux

# Define your model, data, loss, and optimizer
# ...

# Setup learner with quantities
quantities = [LossQuantity(), GradNormQuantity(), DistanceQuantity()]
learner = Learner(model, data_loader, loss_fn, optim, quantities)

# Train with live plotting
train!(learner; epochs=10, with_plots=true, track_every=1)
```
