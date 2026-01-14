# LMD4MLTraining

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://LJS42.github.io/LMD4MLTraining.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://LJS42.github.io/LMD4MLTraining.jl/dev/)
[![Build Status](https://github.com/LJS42/LMD4MLTraining.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/LJS42/LMD4MLTraining.jl/actions/workflows/CI.yml?query=branch%3Amain)
# Live-Monitoring-Debugging-of-ML-Training
LMD4MLTraining.jl is a Julia package for live monitoring and visual debugging of neural network training in Flux.jl.

The package is inspired by the Python package cockpit and aims to provide insight into training dynamics by visualizing diagnostic quantities while training is running.

## Motivation
When training neural networks, issues such as unstable optimization, exploding gradients, or stalled learning are often only detected after training has finished. This package addresses this problem by providing live, interactive visualizations of important training metrics.

## Fartures 
Currently implemented features and features still in development include:

- Integration with standard Flux/Zygote training loops
- Live visualization using Makie.jl
- Monitoring of loss user defined quantities: gradient norm, distance, update size, norm test and gradient history.
- Modular design for adding additional quantities and visual instruments

## Usage

```
julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=. ./examples/mnist.jl
```
