"""cretae a struct to contain the loss (subtype from AbstractQuantity)
set dictionary key and method to compute the current loss (no processing
needed, just return the loss value)"""

struct LossQuantity <: AbstractQuantity
    key::Symbol # lightweight, immutable identifier, starts with : (e.g. :loss)
    LossQuantity() = new(:loss)
end

quantity_key(q::LossQuantity) = q.key

function compute!(q::LossQuantity, loss, grads)
    return loss 
end                     
