# Visualization

This page describes the visualization components of the package.

---

## Dashboard

The dashboard displays multiple plots that update live during training.

Currently implemented plots:
- **Training loss** vs Iteration
- **Parameter distance** (L2 distance from initial weights)
- **Parameter update size** (step-to-step distance)
- **Gradient norm**
- **Gradient norm test** (measuring signal-to-noise)
- **Gradient distribution** (1D histogram)

The dashboard automatically links the X-axis (Iteration) for all relevant plots, ensuring synchronized visualization of the training progress.

---

## Makie integration

The package uses Makie.jl and observable variables to enable live updates of plots
as training progresses.
