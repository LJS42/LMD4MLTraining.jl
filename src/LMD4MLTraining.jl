module LMD4MLTraining

using Flux: Flux
using Zygote: Zygote
using Makie:
    Makie,
    Figure,
    Axis,
    Label,
    Box,
    GridLayout,
    Relative,
    Theme,
    set_theme!,
    lines!,
    linkxaxes!,
    autolimits!,
    hidespines!,
    hidexdecorations!,
    RGBf,
    RGBAf,
    Point2f,
    Observable,
    Top,
    rowgap!,
    colgap!,
    rowsize!,
    colsize!
using WGLMakie: WGLMakie
using Bonito: Bonito, App, Page, DOM, display
using Dates: Dates
using Statistics: Statistics, mean
using Distributed:
    Distributed,
    @spawnat,
    addprocs,
    rmprocs,
    workers,
    RemoteChannel,
    fetch
using Base.Threads: @async
using Logging: Logging, @info, @error
using Sockets: IPv4, listen, getsockname

# Core cockpit and quantities
include("quantities/quantity.jl")
include("quantities/loss.jl")
include("quantities/gradnorm.jl")
include("quantities/distance.jl")
include("quantities/updatesize.jl")
include("quantities/normtest.jl")
include("quantities/gradhist1d.jl")
include("learner.jl")
include("instruments/dashboard.jl")
include("instruments/renderer.jl")

export AbstractQuantity,
    Learner,
    LossQuantity,
    GradNormQuantity,
    DistanceQuantity,
    UpdateSizeQuantity,
    NormTestQuantity,
    GradHist1dQuantity,
    train!,
    compute,
    quantity_key
end