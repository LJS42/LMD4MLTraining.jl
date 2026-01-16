"""
   NormTestQuantity
"""
struct GradNormQuantity <: AbstractQuantity end

_norm_sq(x::AbstractArray{<:Number}) = sum(abs2, x)
_norm_sq(x::Union{Tuple,NamedTuple}) = sum(_norm_sq, values(x))
_norm_sq(x::Nothing) = 0.0
_norm_sq(x) =
    throw(ArgumentError("Unsupported grad leaf type: $(typeof(x))"))

function compute(q::GradNormQuantity, losses, back, grads, params)
    return sqrt(_norm_sq(grads))
end
