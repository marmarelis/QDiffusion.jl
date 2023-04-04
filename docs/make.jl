push!(LOAD_PATH, "../src/")

using QDiffusion
using Documenter

makedocs(sitename="Documentation · QDiffusion.jl")

deploydocs(repo="github.com/marmarelis/QDiffusion.jl.git") # pushed to the special gh-pages branch to be hosted on github.io
