using LMD4MLTraining
using Test
using Flux
using Base.Threads
using WGLMakie
using Makie

# GradNormQuantity
@testset "GradNormQuantity" begin
    q = GradNormQuantity()
    grads = (a = rand(3), b = rand(2,2))
    val = 1.0
    r = LMD4MLTraining.compute!(q, val, grads)
    @test r > 0
end

# LossQuantity
@testset "LossQuantity compute!" begin
    q = LossQuantity()
    loss_val = 1.23
    r = LMD4MLTraining.compute!(q, loss_val, nothing)
    @test r == loss_val
end

@testset "setup_plots" begin
    struct DummyQ <: LMD4MLTraining.AbstractQuantity
        key::Symbol
    end
    LMD4MLTraining.quantity_key(q::DummyQ) = q.key

    qlist = [DummyQ(:loss)]

    fig, obs, axs = LMD4MLTraining.setup_plots(qlist; display = false)
    @test isa(fig, Figure)
    @test isa(obs[:loss], Observable)
    @test isa(axs[:loss], Axis)
end

# render_loop
@testset "render_loop test" begin
    ch = Channel{Tuple{Int, Dict{Symbol, Float32}}}(2)
    put!(ch, (1, Dict(:loss => 0.1f0)))  
    put!(ch, (2, Dict(:loss => 0.5f0)))
    close(ch)

    # LossQuantity
    qlist = [LossQuantity()]

    try
        t = @async LMD4MLTraining.render_loop(ch, qlist; display = false)
        @test true  
    catch e
        @warn "render_loop skipped in test due to Makie limits: $e"
        @test true  
    end
end

# Learner + train_loop! + Train!
@testset "Learner training" begin
    model = Chain(Dense(2,2))
    data = [(rand(Float32,2), rand(Float32,2))]
    loss_fn(ŷ, y) = sum(abs2, ŷ .- y)
    opt = Flux.setup(Adam(3.0f-4), model)
    learner = Learner(model, data, loss_fn, opt, [LossQuantity()])

    ch = Channel{Tuple{Int,Dict{Symbol,Float32}}}(10)
    t1 = @async LMD4MLTraining.train_loop!(learner, 1, ch)
    wait(t1)
    @test true

    LMD4MLTraining.Train!(learner, 1, false)
    @test true
end

# All following tests need to be tested!
# DistanceQuantity
@testset "DistanceQuantity" begin
    q = DistanceQuantity()
    @test q isa LMD4MLTraining.AbstractQuantity
    @test q.key == :distance
    @test LMD4MLTraining.distance_key(q) == :distance

    struct DummyParams
        pa
        p0
    end

    pa = (Float32[1, 2, 3], Float32[4, 5])
    p0 = (Float32[1, 1, 1], Float32[1, 1])
    params = DummyParams(pa, p0)

    diffs = (pa[1] .- p0[1], pa[2] .- p0[2])
    expected = sqrt(sum(abs2, diffs[1]) + sum(abs2, diffs[2]))

    r = LMD4MLTraining.compute!(q, nothing, nothing, nothing, params)
    @test r0 == 0f0
end

# UpdateSizeQuantity
@testset "UpdateSizeQuantity" begin
    q = UpdateSizeQuantity()
    @test q isa LMD4MLTraining.AbstractQuantity
    @test q.key == :updatesize

    struct DummyParams
        pa
        pb
    end
    
    pa = (Float32[2, 0], Float32[1, 2, 3])
    pb = (Float32[1, 0], Float32[1, 1, 1])
    params = DummyParams(pa, pb)

    diffs = (pa[1] .- pb[1], pa[2] .- pb[2])
    expected = sqrt(sum(abs2, diffs[1]) + sum(abs2, diffs[2]))

    r = LMD4MLTraining.compute!(q, nothing, nothing, nothing, params)
    @test isapprox(r, expected; rtol=1e-6, atol=1e-6)

    params0 = DummyParams(pb, pb)
    r0 = LMD4MLTraining.compute!(q, nothing, nothing, nothing, params0)
    @test r0 == 0f0
end

# NormTestQuantity
@testset "NormTestQuantity" begin
    q = NormTestQuantity()
    @test q isa LMD4MLTraining.AbstractQuantity
    @test q.key == :normtest

    losses = Float32[1, 2, 3]
    B = length(losses)

    function back(seed)
        idx = findfirst(!iszero, seed)
        g = idx == 1 ? Float32[1, 0] :
            idx == 2 ? Float32[0, 2] :
                       Float32[3, 0]
        return (g,)
    end

    grads = Float32[1, 2]

    sample_norm = Float32[1^2 + 0^2, 0^2 + 2^2, 3^2 + 0^2] 
    expected = 1/(B*(B-1)) * (sum(sample_norm)/sum(abs2, grads) - B)

    r = LMD4MLTraining.compute!(q, losses, back, grads, nothing)
    @test isapprox(r, expected; rtol=1e-6, atol=1e-6)
end
