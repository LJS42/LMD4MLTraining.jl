using LMD4MLTraining
using Test

@testset "LMD4MLTraining.jl" begin
    include("test_utils.jl")
    include("test_quantities.jl")
    include("test_learner.jl")
    include("test_dashboard.jl")
end
