using LMD4MLTraining
using Test

@testset "Utilities" begin
    @testset "_norm_sq" begin
        # Test array
        @test LMD4MLTraining._norm_sq([1.0, 2.0]) == 5.0
        
        # Test NamedTuple
        @test LMD4MLTraining._norm_sq((a=[1.0, 2.0], b=[3.0])) == 14.0
        
        # Test Nothing
        @test LMD4MLTraining._norm_sq(nothing) == 0.0f0
        
        # Test unsupported type
        @test_throws ArgumentError LMD4MLTraining._norm_sq("string")
    end
end
