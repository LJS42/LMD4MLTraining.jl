```@meta
CurrentModule = LMD4MLTraining
```

# LMD4MLTraining

Documentation for [LMD4MLTraining](https://github.com/LJS42/LMD4MLTraining.jl).

```@index
```

# LMD4MLTraining.jl

`LMD4MLTraining.jl` is a Julia package for **live monitoring and visual debugging**
of neural network training in Flux.jl.

The package is inspired by the Python package
[cockpit](https://github.com/f-dangel/cockpit) and aims to provide insight into
training dynamics by visualizing diagnostic quantities *while training is running*.

---

## Motivation

When training neural networks, issues such as unstable optimization, exploding
gradients, or stalled learning are often only detected after training has finished.
This package addresses this problem by providing **live, interactive visualizations**
of important training metrics.

---

## Features

Currently implemented features include:

- Integration with standard Flux.jl training loops
- Live visualization using Makie.jl
- Monitoring of training loss
- Monitoring of gradient norms and distributions
- Monitoring of parameter distances and update sizes
- Modular design for adding additional quantities and visual instruments

---

## Project status

This package is under active development.

---

## Getting Started

To get started, you can run the provided MNIST example. This example demonstrates how to integrate `LMD4MLTraining.jl` into a Flux.jl training loop.

```julia
using LMD4MLTraining
using Flux
using MLDatasets

# 1. Prepare data loader
function get_data_loader()
    preprocess(x, y) = (reshape(x, 28, 28, 1, :), Flux.onehotbatch(y, 0:9))
    x_train_raw, y_train_raw = MLDatasets.MNIST.traindata()
    x_train, y_train = preprocess(x_train_raw, y_train_raw)
    return Flux.DataLoader((x_train, y_train); batchsize=16, shuffle=true)
end

# 2. Define model
model = Chain(
    Conv((5, 5), 1 => 6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(256, 120, relu),
    Dense(120, 84, relu),
    Dense(84, 10),
)

# 3. Setup optimizer and loss
optim = Flux.setup(Adam(3f-4), model)
loss_fn(ŷ, y) = vec(Flux.logitcrossentropy(ŷ, y; agg=identity))

# 4. Define quantities to track
quantities = [
    LossQuantity(), 
    GradNormQuantity(), 
    DistanceQuantity(), 
    UpdateSizeQuantity(), 
    NormTestQuantity()
]

# 5. Create Learner and start training with plots
learner = Learner(model, get_data_loader(), loss_fn, optim, quantities)
train!(learner, 5, true) # Train for 5 epochs with live plotting
```

When you run this code, a browser window will automatically open with a live dashboard showing the training progress.

## Documentation overview

- **Getting Started**: how to run the MNIST example and open the live dashboard
- **Architecture**: overview of the internal design and module structure
- **Quantities**: description of tracked training quantities
- **Visualization**: dashboard and plotting design
- **API Reference**: exported types and functions
