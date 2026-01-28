# Getting Started

This page shows how to install the package and run `LMD4MLTraining.jl` on a small MNIST example to visualize training dynamics in real time.

---

## Requirements

- Julia
- A working Makie backend

---

## Get the code
Start a Julia REPL and add the package to the desired environment via: 

```bash
julia> using Pkg
julia> Pkg.add(url="https://github.com/LJS42/LMD4MLTraining.jl")
```
Now you can load and use the package: 

```bash
julia> using LMD4MLTraining
```
If instead you want to clone the repository to a desired directory:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
```
Then activate the project and install dependencies: 

```julia
pkg> activate .
pkg> instantiate
```
---

## Quick Start

You can run the provided MNIST example to see the dashboard in action without cloning the repository. Copy and paste the following command into your terminal:

```bash
julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.add([Pkg.PackageSpec(url="https://github.com/LJS42/LMD4MLTraining.jl"), "Flux", "MLDatasets"]); include(download("https://raw.githubusercontent.com/LJS42/LMD4MLTraining.jl/main/examples/mnist.jl"))'
```

If you have already cloned the repository, you can run it using:

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate(); include("examples/mnist.jl")'
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


## MNIST Live Monitoring Example

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
quantities = LMD4MLTraining.AbstractQuantity[
    LossQuantity(), 
    GradNormQuantity(), 
    DistanceQuantity(), 
    UpdateSizeQuantity(), 
    GradNormQuantity()]
#quantities = LMD4MLTraining.AbstractQuantity[] -> if no quantities selected, loss quantity will be plotted

# 5. Create Learner and start training with plots
learner = Learner(model, get_data_loader(), loss_fn, optim, quantities)
train!(learner, 5, true, 20) # Train for 5 epochs with live plotting every 20th step
```

When you run this code, a browser window will automatically open with a live dashboard showing the training progress.
