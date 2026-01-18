"""
    AbstractQuantity
Abstract base type for all quantities tracked during training.
"""
abstract type AbstractQuantity end

"""
    compute(q::AbstractQuantity, losses, back, grads, params)
Compute the value of quantity `q` using the provided training information.
"""
function compute end

"""
    quantity_key(q::AbstractQuantity)
Return a symbol key uniquely identifying the quantity.
"""
function quantity_key end

# Internal utilities
_norm_sq(x::AbstractArray{<:Number}) = sum(abs2, x)
_norm_sq(x::Union{Tuple,NamedTuple,Base.Iterators.Pairs}) = sum(_norm_sq, values(x))
_norm_sq(x::Nothing) = 0.0f0
_norm_sq(x) = throw(ArgumentError("Unsupported grad leaf type: $(typeof(x))"))