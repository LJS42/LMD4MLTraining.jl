using Flux
"""simple Flux training loop that passes loss and gradients to active cockpit session 
"""
#in workflow: 
# data_loader = get_data()
# model = get_model()
# loss_fn(ŷ, y) = Flux.logitcrossentropy(ŷ, y)
# optim = Flux.setup(Adam(3.0f-4), model)

function train_with_cockpit(model, 
                            data_loader, 
                            opt,
                            session::Session; 
                            loss_fn,
                            epochs=1)
    for epoch in 1:epochs
        for (x, y) in data_loader
            loss, grads = Flux.withgradient(m -> loss_fn(m(x),y), model)
            update!(session, loss, grads)
            #alternative: loss, grads = track!(session, q, model, x, y; loss_fn)
            Flux.Optimise.update!(opt, model, grads)
            
        end
        @info "Epoch $epoch complete"
    end 
    return session
    #alternative: no return
end

