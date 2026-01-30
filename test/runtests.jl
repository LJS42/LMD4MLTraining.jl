using LMD4MLTraining
using Test

@testset "LMD4MLTraining.jl" begin
    include("aqua.jl")
    include("test_utils.jl")
    include("test_quantities.jl")
    include("test_learner.jl")
    include("test_learner_validation.jl")
    include("test_dashboard.jl")
end
