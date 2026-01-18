# Quantities

Quantities are numerical values computed during training and logged for visualization.

## Implemented quantities

- `LossQuantity`: Tracks the training loss.
- `GradNormQuantity`: Tracks the L2 norm of the gradients.
- `DistanceQuantity`: Tracks the distance from the initial parameters.
- `UpdateSizeQuantity`: Tracks the size of the parameter updates.
- `NormTestQuantity`: Tracks the norm test for gradient stability.
- `GradHist1dQuantity`: Tracks a histogram of gradient elements.

## Creating new quantities

To create a new quantity, subtype `AbstractQuantity` and implement the following functions:
- `quantity_key(q::MyQuantity)`: returns a unique symbol key.
- `compute(q::MyQuantity, losses, back, grads, params)`: returns the computed value.
- Add it to the dashboard layout in `src/instruments/dashboard.jl`.
