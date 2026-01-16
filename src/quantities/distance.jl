"""
    DistanceQuantity
how far has the network moved in the parameter space since the start of training (L2-Norm)
"""
struct DistanceQuantity <: AbstractQuantity end

function compute(q::DistanceQuantity, losses, back, grads, params)
    params_diff = (a .- b for (a,b) in zip(params.pa, params.p0)) |> Tuple
    return sqrt(_norm_sq(params_diff))
end
