using LMD4MLTraining
using Test
using Flux

@testset "Quantities" begin
    # Setup dummy data for testing quantities
    model = Chain(Dense(2, 1))
    x = randn(Float32, 2, 4)
    y = randn(Float32, 1, 4)
    params0 = [copy(p) for p in Flux.trainables(model)]
    
    # Forward and backward pass
    losses, back = Flux.pullback(m -> vec(Flux.mse(m(x), y; agg=identity)), model)
    grads = back(fill(1.0f0, length(losses)))[1]
    
    params_before = [copy(p) for p in Flux.trainables(model)]
    Flux.update!(Flux.setup(Adam(), model), model, grads)
    params_after = [copy(p) for p in Flux.trainables(model)]
    
    params = (p0=params0, pb=params_before, pa=params_after)

    @testset "LossQuantity" begin
        q = LossQuantity()
        @test quantity_key(q) == :loss
        val = compute(q, losses, back, grads, params)
        @test val isa Float32
        @test val >= 0
        @test isapprox(val, sum(losses)/length(losses))
    end

    @testset "GradNormQuantity" begin
        q = GradNormQuantity()
        @test quantity_key(q) == :gradnorm
        val = compute(q, losses, back, grads, params)
        @test val isa Float32
        @test val >= 0
    end

    @testset "DistanceQuantity" begin
        q = DistanceQuantity()
        @test quantity_key(q) == :distance
        val = compute(q, losses, back, grads, params)
        @test val isa Float32
        @test val >= 0
        # Distance from p0 to pa should be > 0 after update
        @test val > 0
    end

    @testset "UpdateSizeQuantity" begin
        q = UpdateSizeQuantity()
        @test quantity_key(q) == :updatesize
        val = compute(q, losses, back, grads, params)
        @test val isa Float32
        @test val >= 0
        # Update size should be > 0 after update
        @test val > 0
    end

    @testset "NormTestQuantity" begin
        q = NormTestQuantity()
        @test quantity_key(q) == :normtest
        val = compute(q, losses, back, grads, params)
        @test val isa Float32
    end

    @testset "GradHist1dQuantity" begin
        q = GradHist1dQuantity()
        @test quantity_key(q) == :gradhist1d
        val = compute(q, losses, back, grads, params)
        @test val isa Dict
        @test haskey(val, :mean_grad)
    end
end
