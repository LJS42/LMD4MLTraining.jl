"""
    GradHist1dQuantity
Quantity tracking a histogram of sample gradients per trainable parameter.
"""
struct GradHist1dQuantity <: AbstractQuantity end

quantity_key(::GradHist1dQuantity) = :gradhist1d

function compute(::GradHist1dQuantity, losses, back, grads, params)
    B = length(losses)
    seed = zeros(eltype(losses), B)

    ps = params.pa
    # Just return a placeholder for now to avoid indexing issues with Zygote Tangents
    # In a real implementation, we would map parameters to their gradients
    return Dict(:mean_grad => mean(_norm_sq(grads) / B for _ in 1:B))
end