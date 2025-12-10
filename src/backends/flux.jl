export flatten_cockpit
function train_with_cockpit(model, data, opt,session::Session; epochs=1)
    for epoch in 1:epochs
        for (x, y) in data

            # Flux training
            grads = Flux.gradient(model) do m
                session.quantities["loss"].lossfn(m, x, y)
            end
            Flux.Optimise.update!(opt, model, grads)

            # Cockpit monitoring
            update_session!(session, model, x, y)
        end
    end
    return session
end

function flatten_cockpit(x)
    return Flux.flatten(x)
end