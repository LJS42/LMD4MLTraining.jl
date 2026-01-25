using LMD4MLTraining
using Flux
using BenchmarkTools
using Profile, ProfileView

# ---- fixed data (do NOT allocate in benchmark) ----
data = [(randn(Float32, 2, 4), randn(Float32, 1, 4)) for _ in 1:2]
channel = Channel{Tuple{Array{Float32,2},Array{Float32,2}}}(32) do ch
    for batch in data
        put!(ch, batch)
    end
end

loss_fn(ŷ, y) = vec(Flux.mse(ŷ, y; agg=identity))

function make_learner()
    model = Chain(Dense(2, 1))
    optim = Flux.setup(Adam(), model)
    quantities = [LossQuantity(), GradNormQuantity()]
    Learner(model, channel, loss_fn, optim, quantities)
end

# warm-up
learner = make_learner()
train!(learner, 10, false)  # 10 steps, verbose=false

println("== Benchmark ==")
@btime begin
    learner = make_learner()
    train!($learner, 10, false)
end

learner = make_learner()
train!(learner, 10, false)  # warm-up
@profview train!(learner, 10, false)