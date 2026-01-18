"""
    LossQuantity
Quantity tracking the training loss.
"""
struct LossQuantity <: AbstractQuantity end

quantity_key(::LossQuantity) = :loss

function compute(::LossQuantity, losses, back, grads, params)
    return Float32(Statistics.mean(losses))
end
