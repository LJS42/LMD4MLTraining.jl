using LMD4MLTraining
using Test
using Makie
using WGLMakie

@testset "Dashboard and Renderer" begin
    @testset "Layout Combinations" begin
        combinations = [
            [LossQuantity()],
            [LossQuantity(), GradNormQuantity()],
            [LossQuantity(), GradNormQuantity(), DistanceQuantity()],
            [LossQuantity(), GradNormQuantity(), DistanceQuantity(), UpdateSizeQuantity()],
            [LossQuantity(), GradNormQuantity(), DistanceQuantity(), UpdateSizeQuantity(), NormTestQuantity()],
            [LossQuantity(), GradNormQuantity(), DistanceQuantity(), UpdateSizeQuantity(), NormTestQuantity(), GradHist1dQuantity()]
        ]
        
        for qs in combinations
            fig, axes_dict = LMD4MLTraining.build_dashboard(qs)
            @test fig isa Figure
            @test !isempty(axes_dict)
            
            observables = LMD4MLTraining._initialize_plots(axes_dict)
            @test length(observables) >= length(qs)
        end
    end

    @testset "Renderer Loop" begin
        quantities = [LossQuantity()]
        fig, axes_dict = LMD4MLTraining.build_dashboard(quantities)
        observables = LMD4MLTraining._initialize_plots(axes_dict)
        
        # Test with empty channel
        ch = Channel{Tuple{Int,Dict{Symbol,Float32}}}(1)
        close(ch)
        @test LMD4MLTraining._render_loop(ch, fig, axes_dict, quantities, observables) === nothing
        
        # Test with data
        ch = Channel{Tuple{Int,Dict{Symbol,Float32}}}(1)
        put!(ch, (1, Dict(:loss => 0.5f0)))
        close(ch)
        LMD4MLTraining._render_loop(ch, fig, axes_dict, quantities, observables)
        @test length(observables[LossQuantity][]) == 1
    end
end
