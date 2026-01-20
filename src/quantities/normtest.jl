"""
    NormTestQuantity
Quantity tracking the normalized gradient noise.
"""
struct NormTestQuantity <: AbstractQuantity end

quantity_key(::NormTestQuantity) = :normtest

function compute(::NormTestQuantity, losses, back, grads, params)
    B = length(losses)
    seed = zeros(eltype(losses), B)
    sample_norm = zeros(Float32, B)
    for i in 1:B
        fill!(seed, 0)
        seed[i] = 1
        (g_i,) = back(seed)
        sample_norm[i] = _norm_sq(g_i)
    end
    return Float32(1/(B*(B-1))*(sum(sample_norm)/_norm_sq(grads)-B))
end
# used pullback method avoid B forward basses -> faster than: 
#  @inbounds for j in 1:B
#            xj = x[:,:,:,j:j]
#            yj = y[:,j:j]
#              _, gj = Flux.withgradient(m -> loss_one(m, xj, yj), model)
#            gj = gj[1] 
#         end
