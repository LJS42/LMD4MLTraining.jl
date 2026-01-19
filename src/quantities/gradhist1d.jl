"""
    GradHist1dQuantity
Histogram of per-sample gradient elements (one value per sample).
"""
struct GradHist1dQuantity <: AbstractQuantity
    key::Symbol
    nbins::Int
    maxval::Float32
    GradHist1dQuantity(; nbins::Int=30, maxval::Real=5.0) = new(:gradhist1d, nbins, Float32(maxval))
end

quantity_key(q::GradHist1dQuantity) = q.key

function _histcounts_1d(x::AbstractVector{<:Real}, nbins::Int, maxval::Real)
    counts = zeros(Float32, nbins)
    w = Float32(maxval) / nbins
    @inbounds for v in x
        vv = Float32(v)
        b = clamp(Int(floor(vv / w)) + 1, 1, nbins)
        counts[b] += 1f0
    end
    return counts
end

function compute(q::GradHist1dQuantity, losses, back, grads, params)
    B = length(losses)
    seed = zeros(eltype(losses), B)
    per_sample = zeros(Float32, B)

    for i in 1:B
        fill!(seed, 0)
        seed[i] = 1
        (g_i,) = back(seed)
        per_sample[i] = Float32(sqrt(_norm_sq(g_i)))
    end

    return _histcounts_1d(per_sample, q.nbins, q.maxval)
end