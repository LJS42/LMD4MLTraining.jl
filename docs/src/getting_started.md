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
train!(learner, 10, true)
```
