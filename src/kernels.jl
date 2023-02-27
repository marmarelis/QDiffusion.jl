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

using Statistics
using NaNMath
using LazyArrays # for efficiency, in place of refactoring things into the closure-pipeline style?
# from some benchmarks (& hinted by their Github intro) I noticed that materializing simple matrix operations
# is actually faster than eager evaluation... is that because it lowers to BLAS code more often?
# how can I chain LazyArray(@~ ...) with more lazy operations before evaluating eagerly? does it always chain automagically?

abstract type Kernel{T<:Real} end

"""
  A q-deformed Gaussian kernel.
  `deformation` stands for ``q``.

  Here and for the vanilla Gaussian kernel, `bandwidth` is carried
  separately and corresponds precisely to ε.

  The power-law decay should exhibit robustness, and breathe some life
  into situations with proximities 1, 1, 1, then 0, 0, 0, 0...
  It should break apart that bifurcation by reaching further into space.
  Neighborhood "gradients" are smoother in high dimensions---there is no
  longer the problem due to dimishing hypersphere volumes.

  In exchange for fatter tails, these deformed kernels trade off some
  mass in the main bulk of the curve, near the origin. For this
  reason they tend to fare better with larger bandwidths, generally.

  ## On `inner_bandwidth`.
  A q-deformed kernel should really be endowed with two bandwidths: an
  inner bandwidth, heuristically a little greater than unit, that
  multiplies the intradimensional scales before terms are q-summed, and
  an outer bandwidth that serves the same purpose as our solo bandwidth.
""" # LaTeX can be formatted inside double backticks, e.g. ``\\alpha=1``
struct DeformedKernel{T<:Real} <: Kernel{T}
  deformation::T # q
  scales::Vector{T} # do not compute on the fly
  inner_bandwidth::T # could simply multiply into the above, teleologically, but I like the separate semantics
  decay_power::T # alpha-decay applied to what I call the q-Euclidean norm (btw, is this a metric norm?)
  do_deform_norm::Bool
  norm_power::T # defaults to 2.0 for the (deformed) Euclidean norm. could differ from the decay_power
end

function scale_kernel(deformation::T, data::AbstractMatrix{T})::Vector{T} where T <: Real
  centroid = mapslices(NaNMath.mean, data, dims=2) # make sure I don't silently & innocuously override Statistics.mean
  power = 3-deformation
  scale = mapslices(NaNMath.mean, abs.(data .- centroid).^power, dims=2) .^ (1/power)
  return scale[:, 1]
end

# OLD: include a factor of 2 (as in 2sigma^2) or (3-q) under the terms that are q-summed!
# we need to arrive at a more methodical approach to scaling those internals.

function DeformedKernel(
    deformation::T, data::AbstractMatrix{T}, scale::Bool;
    inner_bandwidth::T = T(1), norm_power::T = T(2), decay_power::T = norm_power,
    do_deform_norm::Bool = true) where T <: Real
  scales = scale ?
    scale_kernel(deformation, data) : ones(T, size(data, 1))
  DeformedKernel(deformation, scales,
    inner_bandwidth, decay_power, do_deform_norm, norm_power)
end

@inline q_exp(q::T, x::T, floor::T=T(0)) where T <: Real = max(1 + (1-q)*x, floor)^(1 / (1-q)) # simply broadcast q_exp itself

# what would a q-sum look like when we're using different q's for each dimension? is it even possible to arrive at a compact representation?
# can we get the compiler to unravel this recursive implementation? that would make it moderately suitable for GPUs. employ a combination of @inline and Val{N} to facilitate static unrolling/unwinding.
# this setup may take significantly longer to compile, but hopefully the improved runtime is worth it
# also, what's a "generated function"? Would it be pertinent here?
@inline q_sum_static(q::T, x::AbstractMatrix{T}, ::Val{0}) where T <: Real = zero(T)

# julia is more strict with array shapes and indexing so we will impose a single batch dimension always. we could have broadcasted more fancily as well
Base.@propagate_inbounds \
@inline function q_sum_static(
    q::T, x::AbstractMatrix{T}, n_cols::Val{N})::Vector{T} where {T <: Real, N} # can't impose type constraints on N?
  rest = @view x[2:end, :] # should be copy-on-write by default...
  tail = q_sum_static(q, rest, Val(N-1))
  head = BroadcastArray(@view x[1, :]) do h
    isnan(h) ? zero(T) : h # `ifelse.` for broadcasting. impute slices with median???
  end
  head .+ tail .+ (1-q).*head.*tail
end

# compiles a bespoke function on the fly. products are reused when called on the same dimensions. do applications to fewer dimensions get prepared as well? seems like they would
Base.@propagate_inbounds \
function q_sum_static(q::T, x::AbstractMatrix{T})::Vector{T} where T <: Real
  n_dims = size(x, 1) # was I reading the wrong dimension beforehand?
  q_sum_static(q, x, Val(n_dims))
end

# transplanted from cuda-kernels.jl
Base.@propagate_inbounds \
function q_sum_tree(q::T, x::AbstractMatrix{T})::Vector{T} where T <: Real
  n_dims, n_points = size(x)
  intermediate = zeros(T, n_dims)
  map(1:n_points) do point_index
    current_width = n_dims
    # ...
  end
end

# this is the only branch in the path of execution that is ever actually followed
Base.@propagate_inbounds \
function q_sum_dynamic(q::T, x::AbstractMatrix{T})::Vector{T} where T <: Real
  factor = 1 - q
  n_dims, n_points = size(x)
  # optimal cache locality/coherence
  map(1:n_points) do point_index
    slice = @view x[:, point_index]
    foldl(slice, init=T(0)) do acc, value
      value = isnan(value) ? T(0) : value
      value + acc + factor*value*acc
    end
  end
end

using SparseArrays
# newer code, revamped practices...
# WOW THIS IS INSANELY FAST!!! and much more memory-efficient than the above...
Base.@propagate_inbounds \
function q_sum_dynamic(q::T, x::SparseMatrixCSC{T, Int})::Vector{T} where T <: Real
  factor = 1 - q
  rows, vals = rowvals(x), nonzeros(x)
  n_dims, n_points = size(x)
  map(1:n_points) do point_index
    dim_indices = nzrange(x, point_index)
    foldl(dim_indices, init=T(0)) do acc, index
      item = vals[index]
      item + acc + factor*item*acc
    end
  end
end


# size of x can scale by n_points^2 * n_dims, so be careful with memory. @view helped massively here
# note that the q-sum may be specialized for sparse matrices of the form (n_dims x n_points)
# by iterating through the nonzero entries only, since zero terms in a q-sum make no impact
function q_sum(q::T, x::AbstractMatrix{T})::Vector{T} where T <: Real
  @inbounds q_sum_dynamic(q, x) # propagate this `@inbounds` through calls, though with modern branch prediction this may not matter much
end

"""
  `displacements`: (n_dims x batch)
"""
function evaluate(kernel::DeformedKernel{T}, bandwidth::Union{T, Vector{T}},
    displacements::AbstractMatrix{T}, local_scales::Union{AbstractVector{T}, T}, # like a bandwidth but applied before the q-sum
    point_indices::Union{AbstractVector{Tuple{Int, Int}}, Nothing})::Vector{T} where T <: Real
  q = kernel.deformation
  scales = kernel.scales * kernel.inner_bandwidth
  scaled_displacements = LazyArray(@~ (displacements ./ scales) ./ local_scales' )
  norm = kernel.norm_power
  squared_displacements = abs.(scaled_displacements) .^ norm |> materialize # or keep it lazy throughout the q-sum ?
  if kernel.do_deform_norm # rely on type stability. I can accomplish a true compile-time switch by dispatching to different methods
    squared_norm = q_sum(q, -squared_displacements) # negative q-norm
  else
    squared_norm::Vector{T} = -dropdims(sum(squared_displacements, dims=1), dims=1) # [1, :]
  end
  power = kernel.decay_power / norm
  relative_bandwidth = bandwidth / (kernel.inner_bandwidth ^ norm) # to prevent double-scaling by inner and outer bandwidths
  scaled_norm = LazyArray(@~ .-squared_norm ./ relative_bandwidth )
  q_exp.(q, - scaled_norm .^ power) # (3-q)*bandwidth ?
end

"""
  Contrary to the (mostly still relevant) name, you can adjust the
  exponent through `kernel.norm_power`.
"""
function get_distance_squared(
    kernel::DeformedKernel{T}, displacement::AbstractVector{T})::T where T <: Real
  scales = kernel.scales * kernel.inner_bandwidth
  norm = kernel.norm_power
  ## okay that we modify in place?
  #displacement[:] .= .-abs.(displacement ./ scales) .^ norm
  # I found that each time a LazyArray is accessed, it solely evaluates that entry
  squared_displacement = LazyArray(@~ .- abs.(displacement ./ scales) .^ norm )
  matrix = @view squared_displacement[:, :]
  scaled_distance = -q_sum(kernel.deformation, matrix)[1] # make vector into column
  scaled_distance * (kernel.inner_bandwidth ^ norm)
end


struct GaussianKernel{T<:Real} <: Kernel{T}
  scales::Vector{T}
  decay_power::T # FLEXIBILITY OVERLOAD
  norm_power::T
end

scale_kernel(data::AbstractMatrix{T}) where T <: Real =
  scale_kernel(T(1), data) # as opposed to one(T)

function GaussianKernel(data::AbstractMatrix{T}, scale::Bool;
    norm_power::T = T(2), decay_power::T = norm_power) where T <: Real
  scales = scale ? scale_kernel(data) : ones(T, size(data, 1))
  #data |> scale_kernel |> GaussianKernel
  GaussianKernel(scales, decay_power, norm_power)
end

function evaluate(kernel::GaussianKernel{T}, bandwidth::Union{T, Vector{T}},
    displacements::AbstractMatrix{T}, local_scales::Union{AbstractVector{T}, T},
    point_indices::Union{AbstractVector{Tuple{Int, Int}}, Nothing})::Vector{T} where T <: Real
  #n_evals, n_dims = size(displacements)
  #power = kernel.decay_power / 2
  #results = zeros(T, n_evals)
  ## this code is ugly, but avoids many needless intermediate allocations
  #for i in 1:n_evals
  #  squared_norm = zero(T)
  #  for j in 1:n_dims
  #    scaled_displacement = displacements[i, j] / kernel.scales[j]
  #    squared_displacement = scaled_displacement^2
  #    squared_norm += squared_displacement
  #  end
  #  this_bandwidth = bandwidth isa Vector ? bandwidth[i] : bandwidth
  #  results[i] = exp(-squared_norm^power / this_bandwidth)
  #end
  #results
  scaled_displacements = (displacements ./ kernel.scales) ./ local_scales'
  squared_displacements = abs.(scaled_displacements) .^ kernel.norm_power
  squared_norm = dropdims(sum(
      ifelse.(isnan.(squared_displacements), zero(T), squared_displacements),
    dims=1), dims=1)
  power = kernel.decay_power / kernel.norm_power
  return exp.(-(squared_norm ./ bandwidth) .^ power)
end

"""
  For the adaptive kernel to invoke. In the future, I would like to
  harness a q-Euclidean norm, in other words a direct analogy to
  the effective norm that exists in a q-deformed kernel.
"""
function get_distance_squared(
    kernel::GaussianKernel{T}, displacement::AbstractVector{T})::T where T <: Real
  displacement = ifelse.(isnan.(displacement), zero(T), displacement)
  abs.(displacement ./ kernel.scales) .^ kernel.norm_power |> sum
end

function PossiblyDeformedKernel(
    deformation::T, data::AbstractMatrix{T}, scale::Bool;
    norm_power::T = T(2), decay_power::T = norm_power, inner_bandwidth::T = T(1),
    do_deform_norm::Bool = true, epsilon=1e-5) where T <: Real
  #@assert !any(data .|> isnan) # for the future, consider simply dismissing NaN entries as zeros in the q-sum
  if abs(deformation - 1) < epsilon
    GaussianKernel(data, scale; norm_power, decay_power)
  else
    DeformedKernel(deformation, data, scale;
      inner_bandwidth, norm_power, decay_power, do_deform_norm)
  end
end


struct AdaptiveKernel{T, K <: Kernel{T}} <: Kernel{T}
  inner_kernel :: K
  neighbor_rank :: Int # kNN
  nearest_neighbors :: Vector{T} # evaluate(...) may be called in batches and not the entire dataset, so we must extract nearest neighbors by other means
  in_log :: Bool # are we storing log-distances? so far only for GPU code
end

"""
  Initialize on an explicit representation of the data.
"""
function AdaptiveKernel(neighbor_rank::Int, inner_kernel::Kernel{T},
    data::AbstractMatrix{T}, resample_size::Int=0) where T
  #@assert !any(data .|> isnan)
  n_dims, n_points = size(data)
  # draw with replacement?? ``sample(1:n_points, resample_size)``
  resample = ( resample_size > 0 ?
    () -> randperm(n_points)[1:resample_size] :
    () -> Colon() )
  displacements = [ data[:, resample()] .- data[:, j] for j in 1:n_points ]
  scope_size = resample_size > 0 ? resample_size : n_points
  distances = [
      get_distance_squared(inner_kernel, @view displacements[j][:, i])
    for i in 1:scope_size, j in 1:n_points ]
  # create new variable even though it aliases to the one modified in place
  sorted_distances = sort!(distances, dims=1) # sortperm for argsort, if I were to need it
  # add unit to `nearest_neighbor` because the first is guaranteed to be ourselves
  nearest_neighbors = sorted_distances[neighbor_rank+1, :] # bandwidth is squared distance
  AdaptiveKernel(inner_kernel, neighbor_rank, nearest_neighbors, false)
end

using Distances: Euclidean

"""
  Initialize on an implicit representation of the data.
"""
function AdaptiveKernel(neighbor_rank::Int, leaf_size::Int, inner_kernel::Kernel{T}, data::AbstractMatrix{T}) where T
  #@assert !any(data .|> isnan)
  # off-topic: hope that Julia's x^2.0 short-circuits to straightforward squaring. if so, I could furnish a `p` for any Minkowski norm instead
  n_dims, n_points = size(data)
  scales = inner_kernel.scales
  data_means = dropdims(mapslices(NaNMath.mean, data, dims=2), dims=2) # dedups and scaled are going to have means imputed for NaNs. NOT original `data`
  scaled_data = [ # NOTE: previos PHATE results didn't have this part (scaling), but they didn't rescale the PCA'd data anyway
    (isnan(data[i, j]) ? data_means[i] : data[i, j]) / scales[i] for i in 1:n_dims, j in 1:n_points]
  dedup_cols = Vector{T}[] # ordering need not be preserved, actually
  visited_cols = Set{Vector{T}}() # lower allocational burden, but greater computational to linearly search the current matrix each time
  for i in axes(data, 2)
    @views col = ifelse.(isnan.(data[:, i]), data_means, data[:, i])
    if col in visited_cols
      continue
    end
    push!(dedup_cols, col)
    push!(visited_cols, col)
  end
  dedup_data = hcat(dedup_cols...)
  scaled_dedup_data = dedup_data ./ scales
  structure = build_tree(scaled_dedup_data; leaf_size, norm=Euclidean())
  indices, distances = knn(structure.tree, scaled_data, neighbor_rank+1, true) # sorted
  # when there are NaNs, we can still get zero-distance neighbors even after this entire dedup procedure
  # because NaN entries can disregard apparent differences in some dimensions.
  # options: either use imputed distances for scaling too, or establish a minimum neighbor distance.. <-- the latter is being employed now
  nearest_neighbors = [
    get_distance_squared( # use the same norm as in evaluation, so that the ratio is valid and unitless
      inner_kernel, dedup_data[:, i[end]] .- data[:, p])
    for (i, p) in zip(indices, 1:n_points)]
  min_distance = Iterators.filter(x -> x > 0, nearest_neighbors) |> minimum
  nearest_neighbors = ifelse.(nearest_neighbors .== 0, min_distance, nearest_neighbors)
  AdaptiveKernel(inner_kernel, neighbor_rank, nearest_neighbors, false)
end

"""
  Symmetric version as employed by PHATE, "Visualizing structure and transitions in
  high-dimensional biological data" (2019).
"""
function evaluate(wrapper::AdaptiveKernel{T}, bandwidth::Union{T, Vector{T}},
    displacements::AbstractMatrix{T}, local_scales::Union{AbstractVector{T}, T},
    point_indices::Union{AbstractVector{Tuple{Int, Int}}, Nothing})::Vector{T} where T <: Real
  @assert !wrapper.in_log # this path hasn't been implemented here
  if point_indices === nothing
    n_points = size(displacements, 2)
    point_indices = zip(1:n_points, 1:n_points) |> collect
  end
  left_bandwidths =  [ wrapper.nearest_neighbors[i] for (i, _) in point_indices ]
  right_bandwidths = [ wrapper.nearest_neighbors[i] for (_, i) in point_indices ]
  # could do either `left_bandwidths .* bandwidth` or `sqrt.(left_bandwidths) .* local_scales`
  left = evaluate(
    wrapper.inner_kernel, left_bandwidths .* bandwidth, displacements, local_scales, point_indices)
  right = evaluate(
    wrapper.inner_kernel, right_bandwidths .* bandwidth, displacements, local_scales, point_indices)
  0.5(left .+ right)
end


using ProgressMeter

# surprisingly close to what we found empirically with the colon cancer dataset
# for separating the high-dimensional, low-variance process!
function estimate_inner_bandwidth(data::AbstractMatrix{T};
    quant::Float64 = 0.5, robust::Bool = false )::T where T
  # `median` here serves as a more robust `mean` estimator
  estimate = robust ? median : mean
  scatter = @showprogress [
    (data[i, :]' .- data[i, :]) .^ 2 |> estimate |> sqrt
    for i in axes(data, 1) ]
  quantile(scatter, quant)
end