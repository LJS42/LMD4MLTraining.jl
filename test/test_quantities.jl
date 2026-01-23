using LMD4MLTraining
using Test
using Flux

@testset "Quantities" begin

    @testset "_norm_sq Nothing" begin
        x = nothing
        y = LMD4MLTraining._norm_sq(x)
        @test y == 0.0f0
    end
    
    @testset "quantity_key" begin
        @test quantity_key(LossQuantity())      == :loss
        @test quantity_key(GradNormQuantity())  == :gradnorm
        @test quantity_key(DistanceQuantity())  == :distance
        @test quantity_key(UpdateSizeQuantity())== :updatesize
        @test quantity_key(NormTestQuantity())  == :normtest
        @test quantity_key(GradHist1dQuantity())== :gradhist1d
    end

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
        val = compute(q, losses, back, grads, params)
        @test val isa Float32
        @test val >= 0
    end

    @testset "DistanceQuantity" begin
        q = DistanceQuantity()
        val = compute(q, losses, back, grads, params)
        @test val isa Float32
        @test val >= 0
        # Distance from p0 to pa should be > 0 after update
        @test val > 0
    end

    @testset "UpdateSizeQuantity" begin
        q = UpdateSizeQuantity()
        val = compute(q, losses, back, grads, params)
        @test val isa Float32
        @test val >= 0
        # Update size should be > 0 after update
        @test val > 0
    end

    @testset "NormTestQuantity" begin
        q = NormTestQuantity()
        val = compute(q, losses, back, grads, params)
        @test val isa Float32
    end

    @testset "GradHist1dQuantity" begin
        q = GradHist1dQuantity(nbins=30, maxval=5.0)
        hist = compute(q, losses, back, grads, params)
        @test hist isa Vector{Float32}
        @test length(hist) == q.nbins
        @test all(>=(0f0), hist)
        @test sum(hist) > 0f0
    end

end
