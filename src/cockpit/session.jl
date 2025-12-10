mutable struct Session
    data::Vector{Any}   
    output
    Session() = new(Any[], nothing)
end