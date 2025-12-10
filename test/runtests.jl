using Test   
push!(LOAD_PATH, joinpath(@__DIR__, "../src"))
using LMD4MLTraining
using Flux
using Plots

@testset "Minimal CNN Test" begin
    sess = Session()

    model = Chain(
        Conv((3,3), 1=>8, relu),
        MaxPool((2,2)),
        Flux.flatten,
        Dense(8*13*13, 10),   
        softmax
    )

    x = rand(Float32, 28, 28, 1, 4)  
    y = rand(Float32, 10, 4)

    loss_q = LossQuantity()
    track!(sess, loss_q, model, x, y)

    instr = [LossInstrument()]

    dash = Dashboard(sess,instr)
    show_cockpit(dash)

    @test true   
end
