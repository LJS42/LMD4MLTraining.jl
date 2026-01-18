"""
    DistanceQuantity
Quantity tracking the L2 distance of current parameters from the initial parameters.
"""
struct DistanceQuantity <: AbstractQuantity end

quantity_key(::DistanceQuantity) = :distance

function compute(::DistanceQuantity, losses, back, grads, params)
    params_diff = (a .- b for (a,b) in zip(params.pa, params.p0)) |> Tuple
    return Float32(sqrt(_norm_sq(params_diff)))
end
