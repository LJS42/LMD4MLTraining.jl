"""
   GradHist1dQuantity 
historiogram (later in form of bar plot) of sample gradients per trainable parameter at the current iteration 
"""
struct GradHist1dQuantity <: AbstractQuantity end

function compute(q::GradHist1dQuantity, losses, back, grads, params)
    B = length(losses)
    seed = zeros(eltype(losses), B)

    ps = params.pa
    ps_grad = IdDict{Any, Vector{Any}}() #better use more specific type? 
    for p in ps
        ps_grad[p] = [zero(p) for _ in 1:B]
    end
    
    for i in 1:B
        fill!(seed, 0)
        seed[i] = 1
        (g_i,) = back(seed)
        
        for p in ps
            if g_i[p] === nothing
                fill!(ps_grad[p][i], 0)
            else
                ps_grad[p][i] .= g_i[p]
            end
        end
    end
    return ps_grad #think about how to pass to it is easy to plot 
end