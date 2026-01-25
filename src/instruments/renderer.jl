function _pick_free_port()
    sock = listen(IPv4(127, 0, 0, 1), 0)
    addr = getsockname(sock)
    close(sock)
    port = Int(addr[end])
    return port
end

"""
    run_dashboard(channel, ready_channel, quantities)
Initialize and run the dashboard on the current process.
"""
function _run_dashboard(
    channel::Union{Channel,RemoteChannel},
    ready_channel::RemoteChannel,
    quantities::Vector{<:AbstractQuantity},
)
    fig, axes_dict = build_dashboard(quantities)
    observables = _initialize_plots(axes_dict)

    if !haskey(ENV, "CI")
        WGLMakie.activate!(resize_to = :body)

        # Start a server explicitly   
        port = _pick_free_port()
        server = Bonito.Server("0.0.0.0", port)

        app = App() do session
            return DOM.div(
                fig,
                DOM.style(
                    "body { margin: 0 !important; padding: 0 !important; overflow: hidden !important; }",
                ),
                style = "width: 100%; height: 100%;",
            )
        end

        # Route the app to the root
        Bonito.route!(server, "/" => app)
        url = "http://127.0.0.1:$port"

        # Force an initial update to trigger rendering and resizing
        for obs in values(observables)
            notify(obs)
        end
        notify(fig.scene.events.window_area)

        # Signal that we are ready and provide the URL
        put!(ready_channel, url)
        yield()

        try
            return _render_loop(channel, fig, axes_dict, quantities, observables)
        finally
            # Ensure the server is shut down so reruns don't reuse the old dashboard
            try
                close(server)
            catch e
                @warn "Failed to close Bonito server" exception = e
            end
        end


    else
        # In CI, just signal ready with a dummy URL
        put!(ready_channel, "http://localhost:CI")
    end

    _render_loop(channel, fig, axes_dict, quantities, observables)
end

"""
    initialize_plots(axes_dict) -> observables
Initialize line plots on the provided axes and return a dictionary of observables.
"""
function _initialize_plots(axes_dict::Dict{DataType,Axis})
    observables = Dict{DataType,Observable}()
    for (q_type, ax) in axes_dict
        obs = Observable(Point2f[])
        if q_type == CombinedQuantity
            scatter!(ax, obs, color = RGBf(0.54, 0.71, 0.98)) # Blue
        elseif q_type == UpdateSizeOverlay
            scatter!(ax, obs, color = RGBf(0.96, 0.76, 0.91)) # Pink
        elseif q_type == LossQuantity
            lines!(ax, obs, color = RGBf(0.54, 0.71, 0.98)) # Blue
        elseif q_type == DistanceQuantity
            scatter!(ax, obs, color = RGBf(0.98, 0.70, 0.53)) # Peach
        elseif q_type == GradNormQuantity
            lines!(ax, obs, color = RGBf(0.65, 0.89, 0.63)) # Green
        elseif q_type == GradHist1dQuantity
            barplot!(ax, obs, color = RGBf(0.90, 0.90, 0.55)) # Yellow
        else
            lines!(ax, obs, color = RGBf(0.54, 0.71, 0.98)) # Blue default
        end
        observables[q_type] = obs
    end
    return observables
end

"""
    render_loop(channel, fig, axes_dict, quantities, observables)
Consume training updates from `channel` and update the dashboard in real time.
"""
function _render_loop(
    channel::Union{Channel,RemoteChannel},
    fig::Figure,
    axes_dict::Dict{DataType,Axis},
    quantities::Vector{<:AbstractQuantity},
    observables::Dict{DataType,Observable},
)
    quantity_data = Dict{DataType,Vector{Point2f}}(
        q_type => copy(obs[]) for (q_type, obs) in observables
    )

    for (step, received_quantities) in channel
        # Handle shutdown signal
        if step == -1
            break
        end

        for q in quantities
            q_key = quantity_key(q)
            haskey(received_quantities, q_key) || continue
            val = received_quantities[q_key]

            q_type = typeof(q)
            if q_type == DistanceQuantity && haskey(observables, CombinedQuantity)
                push!(quantity_data[CombinedQuantity], Point2f(step, val))
            elseif q_type == UpdateSizeQuantity && haskey(observables, UpdateSizeOverlay)
                push!(quantity_data[UpdateSizeOverlay], Point2f(step, val))
            elseif q_type == GradHist1dQuantity && haskey(observables, q_type)
                nb = length(val)
                mv = Float32(q.maxval)
                w = 2.0f0 * mv / nb
                xs = (-mv + w / 2.0f0) .+ (0:nb-1) .* w
                quantity_data[q_type] = Point2f.(Float32.(xs), Float32.(val))
            elseif haskey(observables, q_type)
                push!(quantity_data[q_type], Point2f(step, val))
            end
        end

        for (q_type, obs) in observables
            obs[] = quantity_data[q_type]
            if haskey(axes_dict, q_type)
                autolimits!(axes_dict[q_type])
            end
        end
        yield()
    end

    # Final update
    for (q_type, obs) in observables
        obs[] = quantity_data[q_type]
        if haskey(axes_dict, q_type)
            autolimits!(axes_dict[q_type])
        end
    end
    yield()
end
