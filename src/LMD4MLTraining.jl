module LMD4MLTraining

using Flux
using Zygote
using WGLMakie
using Bonito
using Makie
using Dates
using Statistics
using Base.Threads
using Loggingexit

# Core cockpit and quantities
include("quantities/quantity.jl")
include("quantities/loss.jl")
include("quantities/gradnorm.jl")
include("quantities/distance.jl")
include("quantities/updatesize.jl")
include("quantities/normtest.jl")
include("learner.jl")
include("instruments/renderer.jl")

export Learner, LossQuantity, GradNormQuantity, DistanceQuantity, UpdateSizeQuantity, NormTestQuantity, Train!
end
