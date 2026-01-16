"""
    UpdateSizeQuantity:
local, per step version of DistanceQuantity
how aggresively is the optimizer moving at the current step
"""
struct UpdateSizeQuantity <: AbstractQuantity end

function compute(q::UpdateSizeQuantity, losses, back, grads, params)
    params_diff = (a .- b for (a,b) in zip(params.pa, params.pb)) |> Tuple
    return sqrt(_norm_sq(params_diff))
end

