"""
    GradHist1dQuantity
Histogram of per-sample gradient elements.
"""
struct GradHist1dQuantity <: AbstractQuantity
    key::Symbol
    nbins::Int
    maxval::Float32
    GradHist1dQuantity(; nbins::Int=100, maxval::Real=0.05) = new(:gradhist1d, nbins, Float32(maxval))
end

quantity_key(q::GradHist1dQuantity) = :gradhist1d

"""
    _histcounts_1d!
Internal. Accumulate a single signed scalar values into a fixed-range 1D histogram.
"""
function _histcounts_1d!(counts::Vector{Float32}, g::Real, nbins::Int, maxval::Real)
    mv = Float32(maxval)
    x = clamp(Float32(g), -mv, mv) 
    u = (x + mv) / (2f0 * mv) 
    b = clamp(Int(floor(u * nbins)) + 1, 1, nbins)
    @inbounds counts[b] += 1f0
    return nothing
end

function compute(q::GradHist1dQuantity, losses, back, grads, params)
    B = length(losses)
    seed = zeros(eltype(losses), B)
    counts = zeros(Float32, q.nbins)

    for i in 1:B
        fill!(seed, 0)
        seed[i] = one(eltype(seed))
        (g_i,) = back(seed)

        for gradlayer in g_i.layers
            gradlayer === nothing && continue

            #look for all trainable parameters in each gradlayer
            for field in values(gradlayer)
                field === nothing && continue
                if field isa AbstractArray
                    @inbounds for g in field
                        _histcounts_1d!(counts, g, q.nbins, q.maxval)
                    end
                else
                    @warn "GradHist1d: unsupported gradient leaf type; only AbstractArray is supported" typeof(field)
                end
            end
        end
    end
    
    return counts
end

