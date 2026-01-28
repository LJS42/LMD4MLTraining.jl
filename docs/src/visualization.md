# Visualization

This page describes the visualization components of the package.

---

## Dashboard

The dashboard provides a live, interactive view of training dynamics and updates continuously while training is running.

Currently implemented plots:
- **Training loss** vs Iteration
- **Parameter distance** (L2 distance from initial weights)
- **Parameter update size** (step-to-step distance)
- **Gradient norm**
- **Gradient norm test** (measuring signal-to-noise)
- **Gradient distribution** (1D histogram)

All plots that use Iteration on the X-axis are automatically linked, ensuring a synchronized view of training progress across metrics. Axis limits are automatically adjusted during training to keep all data visible.

---

## Makie integration

The package uses Makie.jl and observable variables to enable live updates.

## Dashboard execution (internal)

Visualization is handled by a dedicated render loop that updates plots
continuously while training is running. This loop is decoupled from the
optimization process to avoid blocking training.

Training and visualization communicate via a Channel.
At each iteration, the training loop sends:
  - the current step index
  - a dictionary containing the values of the tracked quantities

Scalar quantities are appended to time series plots. Vector-valued quantities (such as gradient histograms) replace the current plot data.

The dashboard is rendered using WGLMakie and served in a
browser via Bonito.


