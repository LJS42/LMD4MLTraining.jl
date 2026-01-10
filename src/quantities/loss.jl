struct LossQuantity <: AbstractQuantity
    key::Symbol 
    LossQuantity() = new(:loss)
end

quantity_key(q::LossQuantity) = q.key

function compute!(q::LossQuantity, losses, back, grads, params)
    return mean(losses)
end
