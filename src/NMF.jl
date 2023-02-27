##  Copyright 2022 Myrl Marmarelis
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.


using NMF # for the nndsvd initialization, which uses RandomizedLinAlg.jl

# for experimenting with "q-diffused (guided/assisted) NMF"

function par_estimate_nmf(kernel::Kernel{T}, points::Matrix{T}, bandwidth::T;
    ranks::Vector{Int}, regularizations::Vector{T}, n_iterations::Int,
    cuda_devices::Vector{Int}, seed::Int, nmf_points::Matrix{T}=points
    )::Matrix{<:NamedTuple} where T <: Real
  @assert CUDA.functional()
  affinities = par_evaluate_pairwise(
    kernel, bandwidth, points, cuda_devices)
  map(Iterators.product(ranks, regularizations)) do (rank, regularization)
    Random.seed!(seed) # make it reproducible along the spectrum of regularizations
    estimate_nmf(nmf_points, Symmetric(affinities);
      rank, regularization, n_iterations)
  end
end

# weights due to Yong-Deok Kim and Seungjin Choi (2009)
function estimate_nmf(points::Matrix{T}, affinities::Symmetric{T}; # does it really have to be symmetric? thinking of directed graph
    rank::Int, regularization::T, n_iterations::Int, scaling=nothing,
    )::NamedTuple where T <: Real
  @assert all( points .>= 0 )
  n_genes, n_cells = size(points)
  if isnothing(scaling)
    scaling = ones(T, n_cells)
  else
    scaling .*= n_cells * sum(scaling)
  end # always sums to `n_cells`
  # basis, and combination weights transposed
  programs, weights = NMF.nndsvd(points, rank) # (m x k), (k x n)
  diagonal = sum(affinities, dims=1)[:] |> Diagonal
  laplacian = diagonal - affinities
  @showprogress for iteration in 1:n_iterations
    # like in Lee and Roy (2021) but not on square matrices
    program_numer = (points .* scaling') * weights'
    program_denom = (programs * weights .* scaling') * weights'
    programs .*= program_numer ./ max.(program_denom, eps(T))
    program_norms = mean(abs2, programs, dims=1)
    weight_norms = mean(scaling' .* weights.^2, dims=2)
    reg_scaling = T(n_genes / n_cells) * sqrt.( program_norms' ./ max.(weight_norms, eps(T)) )
    weight_numer = # the `.* program_norms'` is to make `regularization` a relative value (invariant to arbitrary scalings)
      programs' * (points .* scaling') + regularization * weights * affinities .* reg_scaling
    weight_denom =
      programs' * (programs * weights .* scaling') + regularization * weights * diagonal .* reg_scaling
    weights .*= weight_numer ./ max.(weight_denom, eps(T))
  end
  # there are a bunch of loose degrees of freedom in this particular factorization
  # for instance, you could easily upscale a program and downscale its associated weights
  # regularization helps but it isn't everything ... may want to consider L1-norm.
  # I do L2 now because I think it translates to a better Euclidean interpretation for the weights.
  program_norms = sum(abs2, programs, dims=1) .|> sqrt #sum(programs .^ norm_power, dims=1) .^ (1/norm_power) # all are non-negative!
  programs ./= program_norms
  weights .*= program_norms'
  (; programs, weights, rank, regularization )
end

function par_regress_linear(outputs::Matrix{T},
    kernel::Kernel{T}, points::Matrix{T}, bandwidth::T;
    regularizations::Vector{T}, cuda_devices::Vector{Int},
    covariates::Matrix{T}=points)::Array{T,3} where T
  @assert CUDA.functional()
  affinities = par_evaluate_pairwise(
    kernel, bandwidth, points, cuda_devices)
  zcat = (x, y) -> cat(x, y, dims=3)
  mapreduce(zcat, regularizations) do regularization
    regress_linaer(outputs, covariates, regularization, affinities)
  end
end

function regress_linear(outputs::Matrix{T}, points::Matrix{T},
    regularization::T, affinities::Symmetric{T})::Matrix{T} where T
  diagonal = sum(affinities, dims=1)[:] |> Diagonal
  laplacian = diagonal - affinities
  regularizer = regularization * points * laplacian * points'
  (outputs * points') / (points * points' + regularizer)
end

# PNMF would be better described as non-negative PCA.
# we're using the implementation from the scPNMF paper.
# todo figure out how to graph-regularize, if we wish
function estimate_pnmf(points::Matrix{T}; rank::Int, n_iterations::Int
    )::Matrix{T} where T <: Real
  @assert all( points .>= 0 )
  (; projection ) = reduce_to_principal_components(points, rank)
  projection = abs.(projection') # (n_genes x rank), initialization step
  n_genes, n_cells = size(points)
  #diagonal = sum(affinities, dims=1)[:] |> Diagonal
  #laplacian = diagonal - affinities
  point_cov = points * points'
  @showprogress for iteration in 1:n_iterations
    proj_numer = 2 .* point_cov * projection
    proj_cov = projection * projection'
    proj_denom = (proj_cov*point_cov + point_cov*proj_cov) * projection
    projection .*= proj_numer ./ max.(proj_denom, eps(T))
  end
  proj_norms = sum(abs2, projection, dims=1) .|> sqrt
  projection ./ proj_norms
end