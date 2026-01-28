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

- Integration with standard Flux/Zygote training loops
- Live visualization using WGLMakie.jl
- Monitoring of user defined quantities: loss, gradient norm, distance, update size, norm test and gradient history.
- Modular design for adding additional quantities and visual instruments

---

## Project status

This package is under active development.

---

## Documentation overview

- **Getting Started**: how to install the package and run the provided example
- **Architecture**: overview of the internal design and module structure
- **Quantities**: description of tracked training quantities
- **Visualization**: dashboard and plotting design
- **API Reference**: exported types and functions
