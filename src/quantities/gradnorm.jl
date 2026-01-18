"""
    GradNormQuantity
Quantity tracking the norm of the model gradients.
"""
struct GradNormQuantity <: AbstractQuantity end

quantity_key(::GradNormQuantity) = :gradnorm

function compute(::GradNormQuantity, losses, back, grads, params)
    return Float32(sqrt(_norm_sq(grads)))
end
