struct Dashboard
    session::Session
    instruments::Vector{LossInstrument}
end

function show_cockpit(d::Dashboard)
    for instr in d.instruments
        plot!(instr, d.session)
    end
end
