"""
    UpdateSizeQuantity
Quantity tracking the L2 distance of parameters before and after the current update step.
"""
struct UpdateSizeQuantity <: AbstractQuantity end

quantity_key(::UpdateSizeQuantity) = :updatesize

function compute(::UpdateSizeQuantity, losses, back, grads, params)
    params_diff = (a .- b for (a, b) in zip(params.pa, params.pb)) |> Tuple
    return Float32(sqrt(_norm_sq(params_diff)))
end
