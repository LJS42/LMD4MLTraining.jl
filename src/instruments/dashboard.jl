struct CombinedQuantity <: AbstractQuantity end
struct UpdateSizeOverlay end

const FIG_BG     = RGBf(0.07, 0.07, 0.11) # Mocha Crust
const AX_LOSS_BG = RGBf(0.12, 0.12, 0.18) # Mocha Base
const AX_BG      = RGBf(0.12, 0.12, 0.18) # Mocha Base

const PN_STEP    = RGBf(0.09, 0.09, 0.15) # Mocha Mantle
const PN_GRADS   = RGBf(0.09, 0.09, 0.15) # Mocha Mantle

const BORDER     = RGBAf(0.35, 0.37, 0.49, 0.5) # Mocha Surface1
const GRID       = RGBAf(0.19, 0.20, 0.27, 0.4) # Mocha Surface0
const TX_MAIN    = RGBf(0.80, 0.84, 0.96) # Mocha Text
const TX_TICKS   = RGBf(0.65, 0.68, 0.78) # Mocha Subtext1

const CLASS_LOSS      = :loss
const CLASS_STEPSIZE  = :stepsize
const CLASS_GRADIENT  = :gradients

plot_class(::AbstractQuantity) = error("class must be defined")
plot_title(::AbstractQuantity) = error("plot_title must be defined")
xlabel(::AbstractQuantity)  = "Iteration"
ylabel(::AbstractQuantity)  = error("ylabel must be defined")
axis_bg(::AbstractQuantity) = AX_BG
n_axes(::AbstractQuantity)  = 1
overlay(::AbstractQuantity) = false

plot_class(::LossQuantity) = CLASS_LOSS
plot_title(::LossQuantity) = "Training loss"
ylabel(::LossQuantity) = "Loss"
axis_bg(::LossQuantity) = AX_LOSS_BG

plot_class(::GradNormQuantity) = CLASS_STEPSIZE
plot_title(::GradNormQuantity) = "Gradient norm"
ylabel(::GradNormQuantity) = "Grad Norm"

plot_class(::DistanceQuantity) = CLASS_STEPSIZE
plot_title(::DistanceQuantity) = "Parameter distance"
ylabel(::DistanceQuantity) = "Distance"

plot_class(::UpdateSizeQuantity) = CLASS_STEPSIZE
plot_title(::UpdateSizeQuantity) = "Parameter update size"
ylabel(::UpdateSizeQuantity) = "Update Size"

plot_class(::NormTestQuantity) = CLASS_GRADIENT
plot_title(::NormTestQuantity) = "Gradient norm test"
ylabel(::NormTestQuantity) = "Norm Test"

plot_class(::GradHist1dQuantity) = CLASS_GRADIENT
plot_title(::GradHist1dQuantity) = "Gradient element historiogram"
xlabel(::GradHist1dQuantity) = "Bin"
ylabel(::GradHist1dQuantity) = "Count"

plot_class(::CombinedQuantity) = CLASS_STEPSIZE
n_axes(::CombinedQuantity) = 2
overlay(::CombinedQuantity) = true

plot_title(::CombinedQuantity) = "Parameter distances"
ylabel(::CombinedQuantity) = ylabel(DistanceQuantity())
overlay_ylabel(::CombinedQuantity) = ylabel(UpdateSizeQuantity())

"""
    _quantities_to_objects(quantites) -> objs
Internal. Converts raw list of quantities to track into dashboard plot objects.
"""
function _quantities_to_objects(qs::Vector{<:AbstractQuantity})
    objs = AbstractQuantity[]
    seen = Set{DataType}()
    for q in qs
        q isa LossQuantity && continue
        T = typeof(q)
        T in seen && continue
        push!(objs, q)
        push!(seen, T)
    end
    has_dist = DistanceQuantity in seen
    has_upd  = UpdateSizeQuantity in seen

    if has_dist && has_upd
        out = AbstractQuantity[]
        inserted = false
        for o in objs
            if o isa DistanceQuantity || o isa UpdateSizeQuantity
                if !inserted
                    push!(out, CombinedQuantity())
                    inserted = true
                end
                continue
            end
            push!(out, o)
        end
        return out
    else
        return objs
    end
end

"""
    _objects_to panels(objs)-> panels
Internal. Group objects in dashboard panels if belonging to the same class (step size or gradients).
"""
function _objects_to_panels(objs::Vector{<:AbstractQuantity})
    steps = [o for o in objs if plot_class(o) == CLASS_STEPSIZE]
    grads = [o for o in objs if plot_class(o) == CLASS_GRADIENT]

    isempty(objs) && return Tuple{Symbol, Vector{<:AbstractQuantity}}[]

    step_priority(o) =
        (o isa DistanceQuantity || o isa UpdateSizeQuantity || o isa CombinedQuantity) ? 0 :
        (o isa GradNormQuantity) ? 1 : 2
   
        grad_priority(o) =
        o isa NormTestQuantity ? 0 :
        o isa GradHist1dQuantity ? 1 : 2

    steps = sort(steps; by=step_priority)
    grads = sort(grads; by=grad_priority)

    panels = Tuple{Symbol, Vector{<:AbstractQuantity}}[]

    !isempty(steps) && push!(panels, (CLASS_STEPSIZE, steps))
    !isempty(grads) && push!(panels, (CLASS_GRADIENT, grads))
    
    return panels
end

function _setup_axis!(ax; bg=AX_BG, overlay=false)
    ax.backgroundcolor = bg
    if overlay
        ax.yaxisposition = :right
        ax.backgroundcolor = RGBAf(0, 0, 0, 0)
        hidespines!(ax, :l, :t, :b)
        hidexdecorations!(ax, grid=false)
        ax.xgridvisible = false
        ax.ygridvisible = false
    end
    return ax
end

function _create_group!(cell, n_objs, title; overlay=false, two_cols=false)
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
    ax1 = nothing
    ax2 = nothing
    ax_overlay = nothing

    if n_objs == 1
        ax1 = _setup_axis!(Axis(cell[3, 2]))
        rowsize!(cell, 2, Relative(0.12))
        rowsize!(cell, 3, Relative(0.76))
        rowsize!(cell, 4, Relative(0.10))

        if overlay
            ax_overlay = _setup_axis!(Axis(cell[3, 2]); overlay=true)
            linkxaxes!(ax1, ax_overlay)
        end
        return ax1, ax2, ax_overlay
    end

    if two_cols
        inner = cell[3, 2] = GridLayout()
        ax1 = _setup_axis!(Axis(inner[1, 1]))
        ax2 = _setup_axis!(Axis(inner[1, 2]))

        rowsize!(cell, 2, Relative(0.12))
        rowsize!(cell, 3, Relative(0.76))
        rowsize!(cell, 4, Relative(0.10))

        colgap!(inner, 10)
        rowsize!(inner, 1, Relative(1.0))
        colsize!(inner, 1, Relative(0.5))
        colsize!(inner, 2, Relative(0.5))

        ax_overlay = nothing
        if overlay
            ax_overlay = _setup_axis!(Axis(inner[1, 1]); overlay=true)
            linkxaxes!(ax1, ax_overlay)
        end
        return ax1, ax2, ax_overlay

    else
        ax1 = _setup_axis!(Axis(cell[2, 2]))
        ax2 = _setup_axis!(Axis(cell[4, 2]))

        rowsize!(cell, 2, Relative(0.445))
        rowsize!(cell, 3, Relative(0.01))
        rowsize!(cell, 4, Relative(0.445))

        if overlay
            ax_overlay = _setup_axis!(Axis(cell[2, 2]); overlay=true)
            linkxaxes!(ax1, ax_overlay)
        end
        return ax1, ax2, ax_overlay
    end
end

function _label_axis!(axes, ax, obj, ax_overlay=nothing)
    ax.title  = plot_title(obj)
    ax.xlabel = xlabel(obj)
    ax.ylabel = ylabel(obj)
    axes[typeof(obj)] = ax

    if obj isa CombinedQuantity && ax_overlay !== nothing
        ax_overlay.ylabel = overlay_ylabel(obj)
        axes[UpdateSizeOverlay] = ax_overlay
    end
end

"""
    build_dashboard(quantities) -> (fig, axes_dict)
Construct the dashboard layout for the given quantities and return the figure and axis mapping. Loss quantity is always plotted
"""
function build_dashboard(qs::Vector{<:AbstractQuantity})
    set_theme!(Theme(
        fontsize = 10,
        Axis = (
            xgridvisible = true, ygridvisible = true,
            xgridcolor = GRID, ygridcolor = GRID,
            xticklabelsize = 9, yticklabelsize = 9,
            xlabelsize = 10, ylabelsize = 10,
            titlesize = 12,
            xtickcolor = TX_TICKS, ytickcolor = TX_TICKS,
            xticklabelcolor = TX_TICKS, yticklabelcolor = TX_TICKS,
            xlabelcolor = TX_MAIN, ylabelcolor = TX_MAIN,
            titlecolor  = TX_MAIN,
        )
    ))
    
    objs = _quantities_to_objects(qs)

    # Use reasonable size that will be scaled by CSS
    f = Figure(backgroundcolor = FIG_BG, size = (1200, 900))
    axes = Dict{DataType, Axis}()

    gd = f[1, 1:2] = GridLayout()
    Label(gd[1, 1:3, Top()], "Live Monitoring of ML Training", fontsize = 20, color = TX_MAIN)

    axL = Axis(gd[2, 2])
    _setup_axis!(axL; bg = axis_bg(LossQuantity()))
    axL.title  = plot_title(LossQuantity())
    axL.xlabel = xlabel(LossQuantity())
    axL.ylabel = ylabel(LossQuantity())
    axes[LossQuantity] = axL

    rowgap!(f.layout, 3)
    colgap!(f.layout, 10)

    rowsize!(gd, 1, Relative(0.01))
    colsize!(gd, 1, Relative(0.15))
    colsize!(gd, 3, Relative(0.15))

    isempty(objs) && return f, axes

    panels = _objects_to_panels(objs)
    gridlayouts = GridLayout[]

    # Collect all axes that share the same X-axis ("Iteration") to link them
    iteration_axes = Axis[axL]

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
    panel_count = length(panels)

    for (gp, (cls, items)) in zip(gridlayouts, panels)
        title = cls == CLASS_STEPSIZE ? "STEP SIZE" : "GRADIENTS"

        needs_overlay = any(overlay, items) 
        n_objs  = min(length(items), 2)  

        two_cols = (panel_count == 1) && (n_objs == 2)
        ax1, ax2, ax_overlay = _create_group!(gp, n_objs, title; overlay=needs_overlay, two_cols=two_cols)

        if n_objs == 1
            _label_axis!(axes, ax1, items[1], ax_overlay)
            if xlabel(items[1]) == "Iteration"
                push!(iteration_axes, ax1)
            end
        else
            _label_axis!(axes, ax1, items[1], (items[1] isa CombinedQuantity) ? ax_overlay : nothing)
            if xlabel(items[1]) == "Iteration"
                push!(iteration_axes, ax1)
            end
            _label_axis!(axes, ax2, items[2])
            if xlabel(items[2]) == "Iteration"
                push!(iteration_axes, ax2)
            end
        end
    end

    if length(iteration_axes) > 1
        linkxaxes!(iteration_axes...)
    end

    return f, axes
end

build_dashboard(qs::AbstractVector) = build_dashboard(AbstractQuantity[qs...])