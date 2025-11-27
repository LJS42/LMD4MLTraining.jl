using LMD4MLTraining
using Documenter

DocMeta.setdocmeta!(LMD4MLTraining, :DocTestSetup, :(using LMD4MLTraining); recursive=true)

makedocs(;
    modules=[LMD4MLTraining],
    authors="Group",
    sitename="LMD4MLTraining.jl",
    format=Documenter.HTML(;
        canonical="https://LJS42.github.io/LMD4MLTraining.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/LJS42/LMD4MLTraining.jl",
    devbranch="main",
)
