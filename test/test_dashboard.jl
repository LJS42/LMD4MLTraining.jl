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
            CombinedQuantity(),
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
            [
                LossQuantity(),
                GradNormQuantity(),
                DistanceQuantity(),
                UpdateSizeQuantity(),
                NormTestQuantity(),
            ],
            [
                LossQuantity(),
                GradNormQuantity(),
                DistanceQuantity(),
                UpdateSizeQuantity(),
                NormTestQuantity(),
                GradHist1dQuantity(),
            ],
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
        ch_empty = Channel{Tuple{Int,Dict{Symbol,LMD4MLTraining.QuantityValue}}}(1)
        close(ch_empty)
        @test LMD4MLTraining._render_loop(
            ch_empty,
            fig,
            axes_dict,
            quantities,
            observables,
        ) === nothing

        # data channel
        ch_data = Channel{Tuple{Int,Dict{Symbol,LMD4MLTraining.QuantityValue}}}(10)
        put!(ch_data, (1, Dict{Symbol,LMD4MLTraining.QuantityValue}(:loss => 0.5f0)))
        put!(ch_data, (2, Dict{Symbol,LMD4MLTraining.QuantityValue}(:loss => 0.8f0)))
        close(ch_data)

        LMD4MLTraining._render_loop(ch_data, fig, axes_dict, quantities, observables)

        @test length(observables[LossQuantity][]) == 2
    end

    @testset "Renderer special branches" begin
        q_hist = GradHist1dQuantity(maxval = 1.0)
        quantities = [LossQuantity(), DistanceQuantity(), UpdateSizeQuantity(), q_hist]

        fig, axes_dict = LMD4MLTraining.build_dashboard(quantities)
        observables = LMD4MLTraining._initialize_plots(axes_dict)

        ch = Channel{Tuple{Int,Dict{Symbol,LMD4MLTraining.QuantityValue}}}(10)

        hist = rand(Float32, 10)
        put!(
            ch,
            (
                1,
                Dict{Symbol,LMD4MLTraining.QuantityValue}(
                    :loss => 0.5f0,
                    :distance => 0.1f0,
                    :updatesize => 0.01f0,
                    :gradhist1d => hist,
                ),
            ),
        )

        hist2 = rand(Float32, 10)
        put!(
            ch,
            (
                2,
                Dict{Symbol,LMD4MLTraining.QuantityValue}(
                    :loss => 0.8f0,
                    :distance => 0.2f0,
                    :updatesize => 0.02f0,
                    :gradhist1d => hist2,
                ),
            ),
        )

        close(ch)

        LMD4MLTraining._render_loop(ch, fig, axes_dict, quantities, observables)

        @test length(observables[LossQuantity][]) == 2
        @test haskey(observables, CombinedQuantity)
        @test length(observables[CombinedQuantity][]) == 2
        @test haskey(observables, LMD4MLTraining.UpdateSizeOverlay)
        @test length(observables[LMD4MLTraining.UpdateSizeOverlay][]) == 2
        @test haskey(observables, GradHist1dQuantity)
        @test length(observables[GradHist1dQuantity][]) == 10
    end


    @testset "_pick_free_port" begin
        port = LMD4MLTraining._pick_free_port()

        @test port isa Int
        @test port > 0
        @test port ≤ 65535
    end

    @testset "GradHist1dQuantity branch" begin
        q = GradHist1dQuantity(maxval = 1.0)
        quantities = [q]

        quantity_data = Dict{DataType,Any}()
        quantity_data[GradHist1dQuantity] = Point2f[]

        observables = Dict(GradHist1dQuantity => Observable(Point2f[]))

        step = 1
        val = rand(Float32, 10)

        q_type = GradHist1dQuantity

        nb = length(val)
        mv = Float32(q.maxval)
        w = 2.0f0 * mv / nb
        xs = (-mv + w / 2.0f0) .+ (0:nb-1) .* w

        quantity_data[q_type] = Point2f.(Float32.(xs), Float32.(val))

        @test length(quantity_data[GradHist1dQuantity]) == nb
    end

end
