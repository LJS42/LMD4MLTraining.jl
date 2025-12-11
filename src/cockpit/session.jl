"""Cockpit session: 
        - step: counter for training steps -> really needed?
        - quantities: each of them must be of subtype AbstractQuantity, 
          each element of the list is an quatity active in the session -> each 
          quantity output one value per step 
        - output: dictionary ontaining the history of the values for each quantity 
          over all training steps 
"""

mutable struct Session
    step::Int
    quantities::Vector{AbstractQuantity}   
    output::Dict{Symbol, Vector{Float64}}

    Session(qs::Vector{AbstractQuantity})= new(0, qs, Dict{Symbol, Vector{Float64}}())
end

function update!(s::Session, loss, grads)
    s.step += 1
    for q in s.quantities 
        key = quantity_key(q)
        value = compute!(q, loss, grads) #new quantity value for current step
        history = get!(s.output, key, Float64[]) #get history output of each quantitiy
        # if there is no history yet, create an empty vector Float64[]
        push!(history, value) #append current value to output history
    end
end
