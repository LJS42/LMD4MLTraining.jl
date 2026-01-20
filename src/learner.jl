"""
    Learner{M, D, F, P, Q}
Object bundling together all information for training.
- `model`: Architecture optimized during training.
- `data_loader`: Iterable for training data.
- `loss_fn`: Function calculating the loss.
- `optim`: Optimizer state.
- `quantities`: Metrics computed every training step.
"""
struct Learner{M, D, F<:Function, P, Q<:Vector{<:AbstractQuantity}}
    model::M
    data_loader::D
    loss_fn::F
    optim::P
    quantities::Q
end

"""
    Learner(model, data_loader, loss_fn, optim)
Convenience constructor for `Learner` without quantities.
"""
function Learner(model, data_loader, loss_fn::Function, optim)
    return Learner(model, data_loader, loss_fn, optim, AbstractQuantity[])
end

"""
    train!(learner, epochs, with_plots)
Train a `Learner` for a number of epochs, optionally with live plotting.
"""
function train!(
    learner::Learner,
    epochs::Int,
    with_plots::Bool,
)
    if with_plots
        # Multi-process setup: Dashboard happens on worker, Training on main
        if length(workers()) == 1 && workers()[1] == 1
            addprocs(1)
        end
        worker_id = workers()[end]

        # Ensure all workers have the package and dependencies loaded
        # We use @eval to avoid top-level macro issues in a function
        @info "Loading LMD4MLTraining on worker $worker_id..."
        Distributed.remotecall_eval(Main, [worker_id], quote
            using LMD4MLTraining
            using Flux
            using WGLMakie
            using Bonito
        end)

        # Remote Channel for cross-process communication
        ch = RemoteChannel(() -> Channel{Tuple{Int,Dict{Symbol,QuantityValue}}}(0))
        # Signal channel to ensure dashboard is ready and get its URL
        ready_ch = RemoteChannel(() -> Channel{String}(1))

        # Start Dashboard on worker process. 
        # We ONLY send the quantities to avoid serializing the whole learner (model, data, etc.). ALWAYS send the loss. 
        if isempty(learner.quantities)
            qs = AbstractQuantity[LossQuantity()]
        elseif any(q -> q isa LossQuantity, learner.quantities)
            qs = learner.quantities
        else
            qs = vcat(AbstractQuantity[LossQuantity()], learner.quantities)
        end
        render_task = @spawnat worker_id begin
            try
                _run_dashboard(ch, ready_ch, qs)
            catch e
                @error "Dashboard Error on worker" exception=(e, catch_backtrace())
                put!(ready_ch, "ERROR")
            end
        end

        # Wait for dashboard to signal it's ready and get the URL
        @info "Waiting for dashboard to be ready..."
        url = take!(ready_ch)
        
        if url == "ERROR"
            @error "Dashboard failed to start on worker"
            return
        end
        
        @info "Dashboard ready at $url, starting training..."

        try
            # Run training on main process
            train_loop!(learner, epochs, ch)
        catch e
            if e isa InterruptException
                @info "Training interrupted by user"
            else
                rethrow(e)
            end
        finally
            put!(ch, (-1, Dict{Symbol,QuantityValue}())) # Signal end
            fetch(render_task)
        end
    else 
        train_loop!(learner, epochs, nothing)
    end
end

"""
    train_loop!(learner, epochs, channel)
Internal training loop that computes quantities and sends them to the display channel.
"""
function train_loop!(
    learner::Learner,
    epochs::Int,
    channel::Union{Channel{Tuple{Int,Dict{Symbol,QuantityValue}}}, RemoteChannel, Nothing}
)
    step_count = 0
    params0 = [copy(p) for p in Flux.trainables(learner.model)]
   
    try
        for epoch in 1:epochs
            for (x, y) in learner.data_loader
                step_count += 1
                params_before = [copy(p) for p in Flux.trainables(learner.model)]

                losses, back = Zygote.pullback(m -> learner.loss_fn(m(x), y), learner.model)
                B = length(losses)
                seed = fill(1 / B, B)
                (grads,) = back(seed)
            
                Flux.update!(learner.optim, learner.model, grads)

                params_after = [copy(p) for p in Flux.trainables(learner.model)]
                params = (p0=params0, pb=params_before, pa=params_after)

                if channel === nothing && step_count < 4
                    l_val = compute(LossQuantity(), losses, back, grads, params)
                    println("STEP ", step_count)
                    println("Loss: ", l_val)
                    gn_val = compute(GradNormQuantity(), losses, back, grads, params)
                    println("Gradient norm: ", gn_val)
                    dist_val = compute(DistanceQuantity(), losses, back, grads, params)
                    println("Distance: ", dist_val)
                    upd_val = compute(UpdateSizeQuantity(), losses, back, grads, params)
                    println("Update size: ", upd_val)
                    nt_val = compute(NormTestQuantity(), losses, back, grads, params)
                    println("Norm test: ", nt_val)
                end
                
                if channel !== nothing
                    computed_quantities = Dict{Symbol,QuantityValue}()

                    # Always include loss
                    loss_q = LossQuantity()
                    computed_quantities[quantity_key(loss_q)] = compute(loss_q, losses, back, grads, params)

                    # Include requested quantities (skip LossQuantity to avoid duplicates)
                    for q in learner.quantities
                        q isa LossQuantity && continue
                        computed_quantities[quantity_key(q)] = compute(q, losses, back, grads, params)
                    end

                    put!(channel, (step_count, computed_quantities))
                    yield()
end
            end
            @info "Epoch $epoch complete"
        end
    catch e
        @error "Training Error" exception = (e, catch_backtrace())
    finally
        if channel !== nothing && !(channel isa RemoteChannel)
            close(channel)
        end
    end
end
