push!(LOAD_PATH, "../src/")

using IntrinsicGeometry
using Documenter

makedocs(sitename="Documentation · IntrinsicGeometry.jl")

deploydocs(repo="github.com/marmarelis/IntrinsicGeometry.jl.git") # pushed to the special gh-pages branch to be hosted on github.io
