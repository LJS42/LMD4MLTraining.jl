"""cretae a struct to contain the gradient norm (subtype from AbstractQuantity)
set dictionary key and method to compute the current gradient norm
gradients from training are passed into the function for computation but not stored"""

struct GradNormQuantity <: AbstractQuantity
    key::Symbol
    GradNormQuantity() = new(:gradnorm)
end

quantity_key(q::GradNormQuantity) = q.key

function compute!(q::GradNormQuantity, loss, grads)
    return sqrt(sum(grads .^2))
end