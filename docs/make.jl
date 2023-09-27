using FastCholesky
using Documenter

DocMeta.setdocmeta!(FastCholesky, :DocTestSetup, :(using FastCholesky); recursive=true)

makedocs(;
    modules=[FastCholesky],
    authors="Bagaev Dmitry <bvdmitri@gmail.com> and contributors",
    repo="https://github.com/biaslab/FastCholesky.jl/blob/{commit}{path}#{line}",
    sitename="FastCholesky.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://biaslab.github.io/FastCholesky.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/biaslab/FastCholesky.jl",
    devbranch="main",
)
