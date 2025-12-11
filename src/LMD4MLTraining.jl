module LMD4MLTraining

using Flux
using Plots

# Core cockpit and quantities 

include("cockpit/quantities/quantity.jl")      # defines AbstractQuantity
include("cockpit/quantities/loss.jl")          # LossQuantity, quantity_key, compute!
include("cockpit/quantities/gradnorm.jl")      # GradNormQuantity, quantity_key, compute!
include("cockpit/session.jl")                  # Session, update!

# Visualization instruments 
include("cockpit/instruments/plot_instruments.jl")  # LossInstrument, plot!(LossInstrument, Session)
include("cockpit/instruments/dashboard.jl")              # Dashboard, show_cockpit

#Backend integration 
include("backends/flux.jl")                   # train_with_cockpit


#Create an empty cockpit Session without any quantities
Session() = Session(AbstractQuantity[])

#alternative for training loop (check flux.jl)
function track!(session::Session,
                q::AbstractQuantity,
                model,
                x,
                y, 
                loss_fn)

    # register quantity if not present
    if !(q in session.quantities)
        push!(session.quantities, q)
    end
    #calculate loss and gradients for current batch and model parameters
    loss, grads = Flux.withgradient(m -> loss_fn(m(x),y), model)
    update!(session, loss, grads)
    
    return session
end

export Session,
       LossQuantity,
       GradNormQuantity,
       track!,
       LossInstrument,
       Dashboard,
       show_cockpit,
       train_with_cockpit

end 