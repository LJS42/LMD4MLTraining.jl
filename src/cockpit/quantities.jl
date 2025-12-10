using Flux
mutable struct LossQuantity
    last_loss::Float64
    LossQuantity() = new(0.0)
end

function update!(q::LossQuantity, loss::Float64)
    q.last_loss = loss
end

function compute_loss(loss::LossQuantity, ŷ, y)
    l = Flux.crossentropy(ŷ, y)  
    loss.last_loss = l
    return l
end