using LMD4MLTraining
using Test
using Makie
using WGLMakie

@testset "Dashboard and Renderer" begin

    @testset "Quantity Plot Properties" begin
        quantities = [
            LossQuantity(),
            GradNormQuantity(),
            DistanceQuantity(),
            UpdateSizeQuantity(),
            NormTestQuantity(),
            GradHist1dQuantity()
        ]

        for q in quantities
            @test LMD4MLTraining.plot_class(q) isa Union{String,Symbol}
            @test LMD4MLTraining.plot_title(q) isa String
            @test LMD4MLTraining.xlabel(q) isa String
            @test LMD4MLTraining.ylabel(q) isa String
            @test LMD4MLTraining.axis_bg(q) !== nothing
            @test LMD4MLTraining.n_axes(q) ≥ 1
            @test LMD4MLTraining.overlay(q) isa Bool
        end
    end


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

        # empty channel
        ch_empty = Channel{Tuple{Int,Dict{Symbol,Float32}}}(1)
        close(ch_empty)
        @test LMD4MLTraining._render_loop(
            ch_empty, fig, axes_dict, quantities, observables
        ) === nothing

        # data channel
        ch_data = Channel{Tuple{Int,Dict{Symbol,Float32}}}(10)
        put!(ch_data, (1, Dict(:loss => 0.5f0)))
        put!(ch_data, (2, Dict(:loss => 0.8f0)))
        close(ch_data)

        LMD4MLTraining._render_loop(
            ch_data, fig, axes_dict, quantities, observables
        )

        @test length(observables[LossQuantity][]) == 2
    end
end