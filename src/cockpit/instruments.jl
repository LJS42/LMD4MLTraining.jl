using Plots

struct LossInstrument end

function plot!(instr::LossInstrument, session::Session)
    steps = sort(collect(keys(session.output)))
    losses = [session.output[s]["LossQuantity"] for s in steps]
    plot(steps, losses, xlabel="Step", ylabel="Loss", title="Loss Curve", legend=false)
end
