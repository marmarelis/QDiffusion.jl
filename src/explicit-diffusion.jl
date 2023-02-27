##  Copyright 2021 Myrl Marmarelis
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

using LinearAlgebra
using Random
using Statistics
using NaNMath

# !!! NOTE !!! Julia stores arrays one column at a time, so I have matrices generally ordered
# as (n_dims x n_points), which is an inversion of my typical style. Embedding coordinates
# stay the other way for now. In the midst of executing a painful switch...
# The benefit here is actually marginal, as the q-sum (for instance) slices a single dimension
# of all points at once.

"""
  For obtaining distances inside the embedding.
"""
function get_pairwise_distances( # could use a fake kernel and pass it through `eval_pairwise_cross_kernel`?
    row_points::AbstractMatrix{T}, col_points::AbstractMatrix{T})::Matrix{T} where T
  @assert size(row_points, 1) == size(col_points, 1)
  n_dims = size(row_points, 1)
  n_rows, n_cols = size(row_points, 2), size(col_points, 2)
  displacements = [ row_points[k, i] - col_points[k, j]
    for k in 1:n_dims, i in 1:n_rows, j in 1:n_cols ]
  distances = sum(displacements.^2, dims=1) .|> sqrt
  distances[1, :, :]
end

function eval_pairwise_cross_kernel(kernel::Kernel{T}, bandwidth::T, row_points::AbstractMatrix{T},
    col_points::AbstractMatrix{T}, reference::Symbol)::Matrix{T} where T
  @assert size(row_points, 1) == size(col_points, 1)
  n_dims = size(row_points, 1)
  n_rows, n_cols = size(row_points, 2), size(col_points, 2)
  n_total = n_rows * n_cols
  displacements = [ row_points[k, i] - col_points[k, j]
    for k in 1:n_dims, i in 1:n_rows, j in 1:n_cols ]
  flat_displacements = reshape(displacements, (n_dims, n_total)) # `reshape` oughtn't copy
  # need to specify all this for the sake of fancier kernels, which might store meta-information on individual points
  if reference == :sym # todo: better names, and consider supplying an enum
    point_indices = [ (i, j) for i in 1:n_rows, j in 1:n_cols ] # |> Iterators.flatten flattens the tuples too
  elseif reference == :row # asymmetric
    point_indices = [ (i, i) for i in 1:n_rows, j in 1:n_cols ]
  elseif reference == :col
    point_indices = [ (j, j) for i in 1:n_rows, j in 1:n_cols ]
  else
    error("argument `reference` must be one of {:sym, :row, :col}")
  end
  point_indices = reshape(point_indices, (n_total,))
  evaluations = evaluate(kernel, bandwidth, flat_displacements, one(T), point_indices)
  return reshape(evaluations, (n_rows, n_cols))
end

function eval_pairwise_kernel(kernel::Kernel{T}, bandwidth::T, points::AbstractMatrix{T})::Matrix{T} where T # Kernel{T} guards T implicitly
  #displacements = [points[i, k] - points[j, k] (FOR MANUAL-TESTING PURPOSES)
  #  for i in 1:size(points, 1), j in 1:size(points, 1), k in 1:size(points, 2)]
  #flat_displacements = reshape(displacements, (size(points, 1)^2, size(points, 2)))
  #evaluations = evaluate(kernel, bandwidth, flat_displacements)
  #return reshape(evaluations, (size(points, 1), size(points, 1)))
  eval_pairwise_cross_kernel(kernel, bandwidth, points, points, :sym)
end

"""
  Assumes that affinities are symmetric.
"""
function normalize_diffusion(affinities::Matrix{T}, alpha::T)::Tuple{Matrix{T}, Matrix{T}} where T
  scale = sum(affinities, dims=1).^alpha
  diffusion = if isapprox(alpha, 0)
    affinities
  else
    affinities ./ (scale .* scale')
  end
  return diffusion, scale # `scale` comes out as a row vector
end

function decompose_diffusion(diffusion::Matrix{T})::Tuple{Vector{T}, Matrix{T}, Matrix{T}} where T
  totals = sum(diffusion, dims=2).^0.5 # symmetric version
  transitions = diffusion ./ (totals .* totals')
  eigval, eigvec = transitions |> Hermitian |> eigen # is this multithreaded? I'm finding 8/16 logical cores running. it is! change with `BLAS.set_num_threads(*)`
  eigval, eigvec, totals
end

# what if I had e.g. type Diffusion{T} = Matrix{T} for stronger type safety?
function embed_diffusion(diffusion::Matrix{T})::Tuple{Vector{T}, Matrix{T}, Vector{T}} where T
  eigval, eigvec, totals = decompose_diffusion(diffusion)
  transition_eigvec = eigvec ./ totals#.^2 divide rows. texture paper doesn't have the square. is this an inconsistency across the papers?
  coordinates = transition_eigvec .* eigval'
  density = dropdims(totals.^2, dims=2)
  return eigval, coordinates, density # last one is pi(x), the approximated kernel-like density
end

"""
  To be pedantic, `manifold_dim` is really the embedding dimensionality.
"""
function estimate_explicit_embedding(kernel::Kernel{T}, points::AbstractMatrix{T};
    bandwidth::T, alpha::T, n_manifold_dims::Int) where T
  @assert 0 <= alpha <= 1
  @assert bandwidth > 0
  @assert n_manifold_dims > 0
  affinities = eval_pairwise_kernel(kernel, bandwidth, points)
  diffusion, _ = normalize_diffusion(affinities, alpha)
  eigens, coords, _ = embed_diffusion(diffusion)
  m = n_manifold_dims
  (eigenvalues = eigens[(end-m):(end-1)],
   coordinates = coords[:, (end-m):(end-1)])
end

"""
  See "Texture separation via a reference set" (2014).
"""
function extend_explicit_embedding(kernel::Kernel{T}, endogenous::AbstractMatrix{T}, # or reference_points, external_points
    exogenous::AbstractMatrix{T}; bandwidth::T, alpha::T, n_manifold_dims::Int) where T # compute the coordinates and eigenvalues over again to ensure harmony at the detriment of some efficiency
  @assert 0 <= alpha <= 1
  @assert bandwidth > 0
  @assert n_manifold_dims > 0
  # need (big x small) matrix. must make it row-stochastic (columns of a row sum to 1) eventually
  cross_affinities = eval_pairwise_cross_kernel(kernel, bandwidth, exogenous, endogenous, :col)
  inner_affinities = eval_pairwise_kernel(kernel, bandwidth, endogenous)
  inner_diffusion, inner_scale = normalize_diffusion(inner_affinities, alpha) # inner_scale is D1 from the paper
  eigenvalues, inner_coords, inner_density = embed_diffusion(inner_diffusion)
  cross_scale = sum(cross_affinities ./ inner_scale, dims=2) # D = SUM_j (a(i,j) / d1(j))
  cross_diffusion = cross_affinities ./ (cross_scale .* inner_scale) # A1 = D^-1 A D1^-1. THIS HAPPENS TO BE ROW-STOCHASTIC
  outer_affinities = cross_affinities * cross_affinities'
  outer_diffusion, _ = normalize_diffusion(outer_affinities, alpha) # huge, so we avoid spectral decomposition on it
  # possible enhancement: allow for a wider kernel (more forgiving) on the exogenous sample?
  outer_density = dropdims(sum(outer_diffusion, dims=2), dims=2) # is this the right place to do it? I'm approximating the larger pi(x). I don't actually use it though.
  proper_embedding_dims = eigenvalues .> 0
  # coordinates are eigenvectors multiplied by the eigenvalues once, so cancel that out. To get commensurate scales, I don't divide by the additional sqrt that the paper uses. We're probably talking about different eigenvalues.
  outer_scale = eigenvalues[proper_embedding_dims]
  outer_coords = cross_diffusion * (inner_coords[:, proper_embedding_dims] ./ outer_scale')
  m = min(n_manifold_dims, size(outer_coords, 2) - 1)
  (eigenvalues = eigenvalues[(end-m):(end-1)],
   inner_coordinates = inner_coords[:, (end-m):(end-1)],
   outer_coordinates = outer_coords[:, (end-m):(end-1)],
   inner_diffusion = inner_diffusion,
   outer_diffusion = outer_diffusion) # this last one is for the Laplace-Beltrami operator
end

# extend from random subset/partition.
# semantics matter: a set of points representing a manifold constitutes a singular embedding
function bootstrap_explicit_embedding(kernel::Kernel{T}, points::AbstractMatrix{T},
    subset::Float64; bandwidth::T, alpha::T, n_manifold_dims::Int) where T
  n_points = size(points, 2)
  n_ref_points = round(Int, subset * n_points)
  ref_indices = randperm(n_points)[1:n_ref_points]
  ref_mask = zeros(Bool, n_points)
  ref_mask[ref_indices] .= true
  ref_points = points[:, ref_mask]
  ext_points = points[:, .!ref_mask]
  embedding = extend_explicit_embedding(kernel, ref_points, ext_points;
    bandwidth, alpha, n_manifold_dims)
  coordinates = zeros(T, n_points, n_manifold_dims)
  m = size(embedding.inner_coordinates, 2)
  coordinates[ref_mask, 1:m] = embedding.inner_coordinates # the rest can remain zero
  coordinates[.!ref_mask, 1:m] = embedding.outer_coordinates
  (eigenvalues = embedding.eigenvalues, coordinates = coordinates)
end

function score_explicit_convergence(score::Function, kernel::Kernel{T}, points::AbstractMatrix{T},
    subset::Float64, n_trials::Int; bandwidth::T, alpha::T, n_manifold_dims::Int)::T where T
  n_points = size(points, 2)
  score_convergence(score, T, n_trials; n_points, n_manifold_dims) do
    bootstrap_explicit_embedding(kernel, points, subset;
      bandwidth, alpha, n_manifold_dims)
  end
end
