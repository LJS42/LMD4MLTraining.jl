using LMD4MLTraining
using Test
using Flux

@testset "Learner" begin
    model = Chain(Dense(2, 1))
    data = [(randn(Float32, 2, 4), randn(Float32, 1, 4)) for _ = 1:2]
    loss_fn(ŷ, y) = vec(Flux.mse(ŷ, y; agg = identity))
    optim = Flux.setup(Adam(), model)
    quantities = [LossQuantity(), GradNormQuantity()]

    @testset "Constructor" begin
        learner = Learner(model, data, loss_fn, optim, quantities)
        @test learner isa Learner
        @test learner.model === model
        @test learner.data_loader === data
        @test learner.loss_fn === loss_fn
        @test learner.optim === optim
        @test learner.quantities === quantities

        # Test convenience constructor
        learner_simple = Learner(model, data, loss_fn, optim)
        @test isempty(learner_simple.quantities)
    end

    @testset "Training" begin
        learner = Learner(model, data, loss_fn, optim, quantities)

        # Test training without plots
        @test LMD4MLTraining.train!(learner, 1, false) === nothing

        # Test with plots (CI mode)
        withenv("CI" => "true") do
            @test LMD4MLTraining.train!(learner, 1, true) === nothing
        end
    end
end
