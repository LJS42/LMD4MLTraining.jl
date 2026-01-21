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
            GradHist1dQuantity(),
            CombinedQuantity()
        ]

        for q in quantities
            # plot_class
            cls = plot_class(q)
            @test cls isa String || cls isa Symbol  # 你的 CLASS_* 定义类型

            # plot_title
            title = plot_title(q)
            @test title isa String

            # xlabel
            xl = xlabel(q)
            @test xl isa String

            # ylabel
            yl = ylabel(q)
            @test yl isa String

            # axis_bg
            bg = axis_bg(q)
            @test bg !== nothing

            # n_axes
            na = n_axes(q)
            @test na ≥ 1

            # overlay
            ov = overlay(q)
            @test ov isa Bool
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

        quantity_data = Dict(q => Point2f[] for q in quantities)

        # Test with empty channel
        ch_empty = Channel{Tuple{Int,Dict{Symbol,Float32}}}(1)
        close(ch_empty)  
        result_empty = LMD4MLTraining._render_loop(ch_empty, fig, axes_dict, quantities, observables)
        @test result_empty === nothing

        # Test with actual data
        ch_data = Channel{Tuple{Int,Dict{Symbol,Float32}}}(10)

        put!(ch_data, (1, Dict(:LossQuantity => 0.5f0)))
        put!(ch_data, (2, Dict(:LossQuantity => 0.8f0)))

        task = @async LMD4MLTraining._render_loop(ch_data, fig, axes_dict, quantities, observables)

        yield()

        close(ch_data)
        wait(task)

        @test length(observables[LossQuantity][]) == 2
    end

end
