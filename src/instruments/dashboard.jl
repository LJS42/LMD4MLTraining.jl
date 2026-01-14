using GLMakie

const FIG_BG     = RGBf(0.07, 0.08, 0.10)
const AX_LOSS_BG = RGBf(0.6, 0.6, 0.6)
const AX_BG      = RGBf(0.8, 0.8, 0.8)

const PN_STEP    = RGBf(0.10, 0.16, 0.24)
const PN_GRADS   = RGBf(0.18, 0.28, 0.38)

const BORDER     = RGBAf(1, 1, 1, 0.20)
const GRID       = RGBAf(0, 0, 0, 0.10)
const TX_MAIN    = RGBf(0.95, 0.96, 0.97)
const TX_TICKS   = RGBf(0.82, 0.84, 0.87)

const CLASS_LOSS      = :loss
const CLASS_STEPSIZE  = :stepsize
const CLASS_GRADIENT  = :gradients

objects = Dict(
    :loss => (
        class = CLASS_LOSS,
        title = "Training Loss", xlabel = "Iterations", ylabel = "Loss",
        axis_bg = AX_LOSS_BG
    ),
    :gradnorm => (
        class = CLASS_STEPSIZE, n_axes = 1,
        title = "Gradient Norm", xlabel = "Iteration", ylabel = "Grad Norm",
        axis_bg = AX_BG
    ),
    :distance => (
        class = CLASS_STEPSIZE, n_axes = 1,
        title = "Parameter Distance", xlabel = "Iteration", ylabel = "Distance",
        axis_bg = AX_BG
    ),
    :updatesize => (
        class = CLASS_STEPSIZE, n_axes = 1,
        title = "Update Size", xlabel = "Iteration", ylabel = "Update Size",
        axis_bg = AX_BG
    ),
    :dist_updatesize => (
        class = CLASS_STEPSIZE, n_axes = 2, overlay = true,
        axis_bg = AX_BG
    ),
    :normtest => (
        class = CLASS_GRADIENT, n_axes = 1,
        title = "Gradient Test", xlabel = "Iteration", ylabel = "Norm Test",
        axis_bg = AX_BG
    ),
    :gradhist1d => (
        class = CLASS_GRADIENT, n_axes = 1,
        title = "Gradient Element Histogram", xlabel = "Gradient Element Value", ylabel = "Frequency",
        axis_bg = AX_BG
    )
)

class_of(o::Symbol) = objects[o].class

function quantities_to_objects(qs::Vector{Symbol})
    quants = Symbol[]
    seen = Set{Symbol}()
    for q in qs
        q == :loss && continue
        q in seen && continue
        push!(quants, q)
        push!(seen, q)
    end
    if (:distance in seen) && (:updatesize in seen)
        out = Symbol[]
        inserted = false
        for q in quants
            if q == :distance || q == :updatesize
                if !inserted
                    push!(out, :dist_updatesize)
                    inserted = true
                end
                continue
            end
            push!(out, q)
        end
        return out
    else
        return quants
    end
end

function objects_to_panels(objs::Vector{Symbol})
    steps = [o for o in objs if class_of(o) == CLASS_STEPSIZE]
    grads = [o for o in objs if class_of(o) == CLASS_GRADIENT]

    step_priority(o) =
        o == :gradnorm ? 2 :
        (o == :distance || o == :updatesize || o == :dist_updatesize) ? 1 : 0
    grad_priority(o) =
        o == :normtest ? 0 :
        o == :gradhist1d ? 1 : 2

    steps = sort(steps; by=step_priority)
    grads = sort(grads; by=grad_priority)

    panels = Vector{Tuple{Symbol, Vector{Symbol}}}()

    if isempty(objs)
        return panels
    end

    has_steps = !isempty(steps)
    has_grads = !isempty(grads)

    if has_steps && has_grads
        push!(panels, (CLASS_STEPSIZE, steps))
        push!(panels, (CLASS_GRADIENT, grads))
        return panels
    end

    if has_steps
        push!(panels, (CLASS_STEPSIZE, steps))
    else
        push!(panels, (CLASS_GRADIENT, grads))
    end
    return panels
end

function setup_axis!(ax::Axis; overlay=false, bg=AX_BG)
    ax.backgroundcolor = bg
    if overlay
        ax.yaxisposition = :right
        ax.backgroundcolor = RGBAf(0, 0, 0, 0)
        hidespines!(ax)
        hidexdecorations!(ax, grid=false)
        ax.xgridvisible = false
        ax.ygridvisible = false
    end
    return ax
end

function create_group(cell, n_objs, title; overlay=false, two_cols=false)
    cell[5, 3] = GridLayout()

    Label(cell[1, 1:3], title;
        fontsize = 13,
        color = TX_MAIN,
        valign = :center
    )
    rowgap!(cell, 6)
    colgap!(cell, 6)

    rowsize!(cell, 1, Relative(0.10))
    rowsize!(cell, 5, Relative(0.02))

    colsize!(cell, 1, Relative(0.01))
    colsize!(cell, 2, Relative(0.98))
    colsize!(cell, 3, Relative(0.01))

    ax_overlay = nothing
    ax2 = nothing

    if n_objs == 1
        rowsize!(cell, 2, Relative(0.12))
        rowsize!(cell, 3, Relative(0.76))
        rowsize!(cell, 4, Relative(0.10))

        ax1 = setup_axis!(Axis(cell[3, 2]))

        if overlay
            ax_overlay = setup_axis!(Axis(cell[3, 2]); overlay=true)
            linkxaxes!(ax1, ax_overlay)
        end

        return ax1, ax2, ax_overlay
    end

    if two_cols
        rowsize!(cell, 2, Relative(0.12))
        rowsize!(cell, 3, Relative(0.76))
        rowsize!(cell, 4, Relative(0.10))

        inner = cell[3, 2] = GridLayout()
        inner[1, 2] = GridLayout()  

        colgap!(inner, 10)
        rowsize!(inner, 1, Relative(1.0))
        colsize!(inner, 1, Relative(0.5))
        colsize!(inner, 2, Relative(0.5))

        ax1 = setup_axis!(Axis(inner[1, 1]))
        ax2 = setup_axis!(Axis(inner[1, 2]))

        if overlay
            ax_overlay = setup_axis!(Axis(inner[1, 1]); overlay=true)
            linkxaxes!(ax1, ax_overlay)
        end

        return ax1, ax2, ax_overlay
    else
        rowsize!(cell, 2, Relative(0.445))
        rowsize!(cell, 3, Relative(0.01))
        rowsize!(cell, 4, Relative(0.445))


        ax1 = setup_axis!(Axis(cell[2, 2]))
        ax2 = setup_axis!(Axis(cell[4, 2]))

        if overlay
            ax_overlay = setup_axis!(Axis(cell[2, 2]); overlay=true)
            linkxaxes!(ax1, ax_overlay)
        end

        return ax1, ax2, ax_overlay
    end
end

function label_axis!(axesdict, ax, o::Symbol; ax_overlay=nothing)
    if o == :dist_updatesize
        ax.title  = objects[:distance].title
        ax.xlabel = objects[:distance].xlabel
        ax.ylabel = objects[:distance].ylabel
        axesdict[:distance] = ax

        if ax_overlay !== nothing
            ax_overlay.ylabel = objects[:updatesize].ylabel
            axesdict[:overlay] = ax_overlay
        end
    else
        ax.title  = objects[o].title
        ax.xlabel = objects[o].xlabel
        ax.ylabel = objects[o].ylabel
        axesdict[o] = ax
    end
end


function build_dashboard(quantities::Vector{<:AbstractQuantity})
    set_theme!(Theme(
        font = "DejaVu Sans",
        fontsize = 10,
        Axis = (
            xgridvisible = true,
            ygridvisible = true,
            xgridcolor = GRID,
            ygridcolor = GRID,
            xticklabelsize = 9,
            yticklabelsize = 9,
            xlabelsize = 10,
            ylabelsize = 10,
            titlesize = 12,
            xtickcolor = TX_TICKS,
            ytickcolor = TX_TICKS,
            xticklabelcolor = TX_TICKS,
            yticklabelcolor = TX_TICKS,
            xlabelcolor = TX_MAIN,
            ylabelcolor = TX_MAIN,
            titlecolor  = TX_MAIN,
        )
    ))
    qs   = Symbol[quantity_key(q) for q in quantities]
    objs = quantities_to_objects(qs)

    f    = Figure(size = (1000, 700), backgroundcolor = FIG_BG)
    axes = Dict{Symbol, Any}()

    gd = f[1, 1:2] = GridLayout()
    Label(gd[1, 1:3, Top()], "Live Monitoring of ML Training", fontsize = 20, color = TX_MAIN)

    axL = Axis(gd[2, 2])
    setup_axis!(axL; bg = AX_LOSS_BG)
    axL.title  = objects[:loss].title
    axL.xlabel = objects[:loss].xlabel
    axL.ylabel = objects[:loss].ylabel
    axes[:loss] = axL

    rowgap!(f.layout, 3)
    colgap!(f.layout, 10)

    rowsize!(gd, 1, Relative(0.01))
    colsize!(gd, 1, Relative(0.15))
    colsize!(gd, 3, Relative(0.15))

    if isempty(objs)
        return f, axes
    end

    panels = objects_to_panels(objs)
    gridlayouts = GridLayout[]

    if length(panels) == 1
        (cls, _) = panels[1]
        panel_color = cls == CLASS_STEPSIZE ? PN_STEP : PN_GRADS

        wrap = f[2, 1:2] = GridLayout()
        wrap[3, 3] = GridLayout()

        rowsize!(wrap, 1, Relative(0.02)); rowsize!(wrap, 2, Relative(0.96)); rowsize!(wrap, 3, Relative(0.02))
        colsize!(wrap, 1, Relative(0.02)); colsize!(wrap, 2, Relative(0.96)); colsize!(wrap, 3, Relative(0.02))

        Box(wrap[1:3, 1:3], color=panel_color, strokecolor=BORDER, strokewidth=0.5, z=-100)

        gp = wrap[2, 2] = GridLayout()
        push!(gridlayouts, gp)
    else
        colsize!(f.layout, 1, Relative(0.5))
        colsize!(f.layout, 2, Relative(0.5))

        (cls1, _) = panels[1]
        (cls2, _) = panels[2]
        col1_color = cls1 == CLASS_STEPSIZE ? PN_STEP : PN_GRADS
        col2_color = cls2 == CLASS_STEPSIZE ? PN_STEP : PN_GRADS

        wrap1 = f[2, 1] = GridLayout()
        wrap2 = f[2, 2] = GridLayout()

        wrap1[3, 3] = GridLayout()
        wrap2[3, 3] = GridLayout()

        rowsize!(wrap1, 1, Relative(0.02)); rowsize!(wrap1, 2, Relative(0.96)); rowsize!(wrap1, 3, Relative(0.02))
        colsize!(wrap1, 1, Relative(0.02)); colsize!(wrap1, 2, Relative(0.96)); colsize!(wrap1, 3, Relative(0.02))

        rowsize!(wrap2, 1, Relative(0.02)); rowsize!(wrap2, 2, Relative(0.96)); rowsize!(wrap2, 3, Relative(0.02))
        colsize!(wrap2, 1, Relative(0.02)); colsize!(wrap2, 2, Relative(0.96)); colsize!(wrap2, 3, Relative(0.02))

        
        Box(wrap1[1:3, 1:3], color=col1_color, strokecolor=BORDER, strokewidth=0.5, z=-100)
        Box(wrap2[1:3, 1:3], color=col2_color, strokecolor=BORDER, strokewidth=0.5, z=-100)
    
        gp1 = wrap1[2, 2] = GridLayout()
        gp2 = wrap2[2, 2] = GridLayout()
        push!(gridlayouts, gp1)
        push!(gridlayouts, gp2)

    end

    rowsize!(f.layout, 2, Relative(0.7))

    for (gp, (cls, objlist)) in zip(gridlayouts, panels)
        title = cls == CLASS_STEPSIZE ? "STEP SIZE" : "GRADIENTS"

        overlay = any(get(objects[o], :overlay, false) for o in objlist)
        n_objs  = min(length(objlist), 2)  

        two_cols = (length(objs) == 2) && (length(objlist) == 2)
        ax1, ax2, ax_overlay = create_group(gp, n_objs, title; overlay=overlay, two_cols=two_cols)

        if n_objs == 1
            label_axis!(axes, ax1, objlist[1]; ax_overlay=ax_overlay)
        else
            o1, o2 = objlist[1], objlist[2]
            if o1 == :dist_updatesize
                label_axis!(axes, ax1, o1; ax_overlay=ax_overlay)
                label_axis!(axes, ax2, o2)
            elseif o2 == :dist_updatesize
                label_axis!(axes, ax2, o2; ax_overlay=ax_overlay)
                label_axis!(axes, ax1, o1)
            else
                label_axis!(axes, ax1, o1)
                label_axis!(axes, ax2, o2)
            end
        end

    end
    return f, axes
end

#Testing 
abstract type AbstractQuantity end
struct LossQuantity       <: AbstractQuantity end
struct GradNormQuantity   <: AbstractQuantity end
struct DistanceQuantity   <: AbstractQuantity end
struct UpdateSizeQuantity <: AbstractQuantity end
struct NormTestQuantity   <: AbstractQuantity end
struct GradHist1dQuantity <: AbstractQuantity end

quantity_key(::LossQuantity)       = :loss
quantity_key(::GradNormQuantity)   = :gradnorm
quantity_key(::DistanceQuantity)   = :distance
quantity_key(::UpdateSizeQuantity) = :updatesize
quantity_key(::NormTestQuantity)   = :normtest
quantity_key(::GradHist1dQuantity) = :gradhist1d

f, axes = build_dashboard([
    LossQuantity(),
    GradNormQuantity(),
    DistanceQuantity(),
    UpdateSizeQuantity(),
    NormTestQuantity(),
    GradHist1dQuantity()   
])

f