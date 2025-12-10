module LMD4MLTraining

# cockpit
include("cockpit/session.jl")
include("cockpit/quantities.jl")
include("cockpit/instruments.jl")
include("cockpit/utils.jl") 

# visualization
include("visualization/plots.jl")
include("visualization/dashboard.jl")

# backend
include("backends/flux.jl")

include("ext.jl")

end
