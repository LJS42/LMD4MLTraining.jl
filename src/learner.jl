"""
    Learner 
Object bundling together all information for training with cockpit. Mutable struct so that model parameters and optimizer state 
can be updated in-place during training. 
Fields:
- model: M
arquitecture which parameters should be optimized during training 
- data_loader: D
iterable that makes data batches accessible for training
- loss_fn: F <: Function
calculate the loss of the model w.t.r to training objective (returns scalar loss value)
- optim: P
optimizer chosen to update model parameters in the backward pass (e.g. Flux.Adam)
- quantities: Q <: Vector{<:AbstractQuantity}
(optional) metrics computed every training step used for evakuation and diagnostic of model training
"""
mutable struct Learner{M, D, F<:Function, P, Q<:Vector{<:AbstractQuantity}}
    model:: M
    data_loader::D
    loss_fn::F
    optim::P
    quantities::Q
end
Learner(model, data_loader, loss_fn::Function, optim) =
    Learner(model, data_loader, loss_fn, optim, AbstractQuantity[])

"""
    train!(learner, epochs, with_plots)
train a Learner and render quantities (optinal). 
when plotting desired: create channel to pass data from training loop to cockpit session:
- use put! to pass data into the channel (if full wait until space availiable, else add immediately),
- use take! to returm data from the channel (if channel is empty wait until data arrives, else retrieve inmediately). 

Args:
learner: Learner
- contains model and model training specifications (architecture, loss function, optimizer, quantities to track, etc.)
epochs: Int
- number of training epochs
with_plots: Bool
- user selection, if rendering is desired with_plots = True
"""
function train_learner!(
    learner::Learner,
    epochs::Int,
    with_plots::Bool,
    )
    if with_plots
        ch = Channel{Tuple{Int,Dict{Symbol,Float32}}}(100)

        train_task = @async train_loop!(learner, epochs, ch)
        render_task = @async render_loop(ch, learner.quantities)

        wait(train_task)
        wait(render_task)
    else 
        train_task = @async train_loop!(learner, epochs, nothing)
        wait(train_task)
    end
    
end

"""
    train_loop!(learner, epochs, channel)
Run training for a Learner and send training quantities through a channel for visualization 
Perform model optimization loop: iteration over epochs and batches, use loss and corresponding gradients w.r.t trainable 
parameters to update the model in-place and compute optinal metrics (quantities).
Use a global step counter for traning steps and a channel that automatically closes if task is finished or an error occurs

Args: 
- learner: Learner,
contains model and model training specifications (architecture, loss function, optimizer, quantities to track, etc.)
- epochs: Int,
number of training epochs
- channel: Channel{Tuple{Int,Dict{Symbol,Float32}}} or nothing
communication channel with capacity set to 100 to pass information between Flux backend and cockpit, needed for plotting
"""
function train_loop!(
    learner::Learner,
    epochs::Int,
    channel::Union{Channel{Tuple{Int,Dict{Symbol,Float32}}}, Nothing}
)
    step_count = 0
    params0 = [copy(p) for p in Flux.trainables(learner.model)]
   
    try
        for epoch in 1:epochs
            for (x, y) in learner.data_loader
                if step_count == 0
                    params0 = [copy(p) for p in Flux.trainables(learner.model)]
                end
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
                    loss = compute!(LossQuantity(), losses, back, grads, params)
                    println("STEP ", step_count)
                    println("Loss: ",loss)
                    gradnorm = compute!(GradNormQuantity(), losses, back, grads, params)
                    println("Gradient norm: ",gradnorm)
                    distance = compute!(DistanceQuantity(), losses, back, grads, params)
                    println("Distance: ",distance)
                    updatesize = compute!(UpdateSizeQuantity(), losses, back, grads, params)
                    println("Update size: ",updatesize)
                    normtest = compute!(NormTestQuantity(), losses, back, grads, params)
                    println("Norm test: ",normtest)
                end
                
                if channel !== nothing
                    computed_quantities = Dict{Symbol,Float32}()
                    for q in learner.quantities
                        value = compute!(q, losses, back, grads, params)
                        computed_quantities[quantity_key(q)] = value
                    end
                    put!(channel, (step_count, computed_quantities))
                    sleep(0.001) # To make concurrency possible
                end
            end
            @info "Epoch $epoch complete"
        end
    catch e
        @error "Training Error" exception = (e, catch_backtrace())
    finally
        if channel !== nothing
            close(channel)
        end
    end
 
end

