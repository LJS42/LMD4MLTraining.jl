using WGLMakie
using Bonito
using Statistics
using Dates
using Makie
using Base.Threads

function setup_plots(
    quantities::Vector{<:AbstractQuantity};
    display::Bool = true,
)
    set_theme!(theme_black())
    fig = Figure(size=(1000, 600), fontsize=18)
    axs = Dict{Symbol,Axis}()
    observables = Dict{Symbol,Observable}()

    for (i, q) in enumerate(quantities)
        key = quantity_key(q)
        ax = Axis(
            fig[i, 1],
            title = string(key),
            xlabel = "Step",
            ylabel = "Value",
            yscale = log10,
        )
        axs[key] = ax
        obs = Observable(Point2f[])
        lines!(ax, obs, color = :cyan)
        observables[key] = obs
    end

    if display && !haskey(ENV, "CI")
        WGLMakie.activate!()
        Bonito.browser_display()
        display(fig)
    end

    return fig, observables, axs
end

function render_loop(
    channel::Channel,
    quantities::Vector{<:AbstractQuantity};
    display::Bool = true,
)
    fig, observables, axs = setup_plots(quantities; display = display)

    quantity_data = Dict{Symbol,Vector{Point2f}}(
        key => copy(obs[]) for (key, obs) in observables
    )

    for (step, received_quantities) in channel
        for (key, value) in received_quantities
            haskey(quantity_data, key) || continue
            push!(quantity_data[key], Point2f(step, value))
        end

        for (key, obs) in observables
            obs[] = quantity_data[key]
        end

        for ax in values(axs)
            autolimits!(ax)
        end
    end
end