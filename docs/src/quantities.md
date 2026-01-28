# Quantities

Quantities are numerical diagnostics computed during training and logged for live visualization.
They are evaluated **once per optimization step** (i.e. per batch).

Below is an overview of what is currently tracked and how to interpret it.

## Implemented quantities

### `LossQuantity()`

Tracks the **mean batch loss**.

- **Definition:** `mean(losses)` where `losses` is the per-sample loss vector returned by your loss function.
- **Interpretation:**Measures how well the model fits the current batch and tells you whether training is making immediate progress. Should generally decrease over time (depending on learning rate schedule, regularization, etc.).

### `GradNormQuantity()`

Tracks the **L2 norm of the gradient** of the batch loss.

- **Definition:** `‖∇θ L‖₂` (computed as `sqrt(sum(abs2, grads)))` across all trainable parameters).
- **Interpretation:**
  - Spikes can indicate exploding gradients or a too-large step size.
  - Very small values can indicate vanishing gradients or stalled learning

### `DistanceQuantity()`

Tracks the **L2 distance from the initialization**.

- **Definition:** `‖θₜ - θ₀‖₂`.
- **Interpretation:**
  - Helps diagnose whether training is actually moving the parameters.
  - Useful for comparing runs: some instabilities show up as rapid parameter drift.

### `UpdateSizeQuantity()`

Tracks the **size of the applied parameter update** in each step.

- **Definition:** `‖θ_after - θ_before‖₂`.
- **Interpretation:**
  - Useful to spot overly aggressive updates and check if the optimizer is still actively moving or has entered a noisy diffusion regime
  
### `NormTestQuantity()`

Tracks a **normalized gradient noise estimate**.

- **Definition (batch size `B`):**
  - Compute per-sample gradients `gᵢ` via a pullback on the per-sample loss vector.
  - Let `g` be the summed batch gradient used for the optimizer step.
  - Report
    
    `1 / (B (B-1)) * ( (∑ᵢ ‖gᵢ‖₂²) / ‖g‖₂² - B )`.

- **Interpretation:**
  - Larger values suggest the batch gradient is dominated by noise/variance across samples. Helps diagnose batch size issues and unstable optimization.
  - Near-zero values suggest per-sample gradients are relatively aligned.

### `GradHist1dQuantity(; nbins=100, maxval=0.05)`

Tracks a **1D histogram of gradient element values**.

- **Parameters:**
  - `nbins`: number of histogram bins.
  - `maxval`: clamp range for gradient elements (values outside `[-maxval, maxval]` are clamped).
- **Interpretation:** helps detect issues like poor data scaling, vanishing/exploding gradients, or dead layers
- **Cost note:** This quantity uses **per-sample pullbacks** and iterates over all gradient arrays, so it can be noticeably more expensive than scalar quantities.

## Creating new quantities

To create a new quantity, subtype `AbstractQuantity` and implement the following functions:
- `quantity_key(q::MyQuantity)`: returns a unique symbol key.
- `compute(q::MyQuantity, losses, back, grads, params)`: returns the computed value.
- Add it to the dashboard layout in `src/instruments/dashboard.jl`.


