# Architecture

This page describes the internal architecture of `LMD4MLTraining.jl`.

---

## Overview

The package is organized into the following components:

- Training backend (Flux integration)
- Quantities (metrics computed during training)
- Visualization (Makie-based dashboard)
- Session management

---

## Module structure

- `quantities/`: defines the `AbstractQuantity` interface and specific metrics (loss, gradients, etc.)
- `instruments/`: contains the `renderer` and `dashboard` for live visualization.
- `learner.jl`: provides the `Learner` abstraction and the training loop integrations.

This modular design allows new quantities and visual instruments to be added
without modifying the core training logic.

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
train!(learner, epochs, with_plots, track_every)
```

- `learner`: Your `Learner` instance.
- `epochs`: Number of training epochs.
- `with_plots`: Boolean. If `true`, starts a WGLMakie dashboard in your browser (or VS Code plot pane).
- `track_every`: Int. Number of steps for which quantities computation should be skipped (allows for speed-up) If 1, every step is tracked.

### Available Quantities

The package provides several diagnostic quantities inspired by the "Cockpit" paper:

| `LossQuantity()` | Tracks the training loss over time. |
| `GradNormQuantity()` | Monitors the L2 norm of the gradients. |
| `DistanceQuantity()` | Measures the L2 distance from the initialization. |
| `UpdateSizeQuantity()` | Tracks the L2 norm of the parameter updates. |
| `NormTestQuantity()` | Computes the "norm test" (checks if the gradient is dominated by noise). |
| `GradHist1dQuantity()` | Visualizes the 1D distribution of gradient elements. |
