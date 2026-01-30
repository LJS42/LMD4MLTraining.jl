using LMD4MLTraining
using Test
using Flux

# Minimal working learner pieces
@testset "Learner validation" begin
    model = Chain(Dense(2, 1))
    opt = Flux.setup(Adam(), model)
    loss_fn(ŷ, y) = vec(Flux.mse(ŷ, y; agg=identity))

    @testset "epochs must be > 0" begin
        data = [(rand(Float32, 2, 4), rand(Float32, 1, 4))]
        learner = Learner(model, data, loss_fn, opt, AbstractQuantity[LossQuantity()])

        @test_throws ArgumentError train!(learner, 0, false, 1)
        @test_throws ArgumentError LMD4MLTraining.train_loop!(learner, 0, nothing, 1)
    end

    @testset "track_every must be > 0" begin
        data = [(rand(Float32, 2, 4), rand(Float32, 1, 4))]
        learner = Learner(model, data, loss_fn, opt, AbstractQuantity[LossQuantity()])

        @test_throws ArgumentError train!(learner, 1, false, 0)
        @test_throws ArgumentError LMD4MLTraining.train_loop!(learner, 1, nothing, 0)
    end

    # Iterate empty data
    @testset "empty data_loader throws" begin
        empty_data = Tuple{Any,Any}[]
        learner = Learner(model, empty_data, loss_fn, opt, AbstractQuantity[LossQuantity()])

        @test_throws ArgumentError train!(learner, 1, false, 1)
        @test_throws ArgumentError LMD4MLTraining.train_loop!(learner, 1, nothing, 1)
    end

    # Model with no trainable params
    @testset "model with no trainables throws" begin
        no_trainable_model = x -> x
        data = [(rand(Float32, 2, 4), rand(Float32, 1, 4))]
        learner = Learner(no_trainable_model, data, loss_fn, opt, AbstractQuantity[LossQuantity()])

        @test_throws ArgumentError train!(learner, 1, false, 1)
        @test_throws ArgumentError LMD4MLTraining.train_loop!(learner, 1, nothing, 1)
    end
end
