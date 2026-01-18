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
