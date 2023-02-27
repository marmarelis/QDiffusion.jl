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

using Distances: Metric, Euclidean, Minkowski, Cityblock
using NearestNeighbors
using LinearAlgebra
using SparseArrays
using DataStructures

## train of thought
# we will partition based on the Euclidean distance, as all considered norms have effective `p` at most 2, the Euclidean
# in other words, the q-norm with q>=1 is always at least as robust as the Euclidean norm
# as proven in my paper, the q-norm is actually lower-bounded by the Euclidean norm even though it increases less rapidly
# around sparse displacements
# weird, since a less robust p-norm (say p+a, a>0) than one for a given p is upper-bounded by the more robust one.
# that is the opposite of what my q-norm does
# in terms of kernels, you can never surpass the power law with an exponential, no matter the norm contained within
# as a consequence of the above, we will employ a Minkowski distance (i.e. arbitrary L^p-norm) with p << 1 to never
# (practically) overestimate a distance. is it a problem with the tree structure for our norm to not actually be a metric?

# to enable fast dot products, we construct a tree (which can be reused for different bandwidths) and then save the
# (index, value) pairs for nonzero kernel evaluations on all the neighbors of each point. effectively, that would
# entail reimplementing sparse matrices, so we use Julia's instead.
# spzeros(T, n, m) -> SparseMatrixCSC{T, Int} these support fast row access within a column, but *not* vice versa
# sparse(I, J, V) creates a matrix such that res[I[k], J[k]] := V[k] much more efficiently than manual repopulation

# sparse map should be a thing.. standard
function spmap(f::Function, matrix::SparseMatrixCSC{T,Int}) where T <: Real
  (I, J, V) = findnz(matrix)
  sparse(I, J, map(f, I, J, V))
end

function impute_points(points::AbstractMatrix{T}, means::Union{T, AbstractVector{T}}
    )::AbstractMatrix{T} where T <: Real
  n_dims, n_points = size(points)
  BroadcastArray(points, reshape(1:n_dims, n_dims, 1)) do p, i
    isnan(p) ? (length(means) == 1 ? means[1] : means[i]) : p
  end
end

function impute_points(points::SparseMatrixCSC{T,Int}, means::Union{T, AbstractVector{T}}
    )::SparseMatrixCSC{T,Int} where T <: Real
  spmap(points) do i, j, v
    isnan(v) ? (length(means) == 1 ? means[1] : means[i]) : v
  end
end

# TODO/IDEA: scale dimensions before evaluating this norm that approximates the true one we use?
function build_tree(points::AbstractMatrix{T}...;
    leaf_size::Int, norm::Metric = Cityblock(),
    impute_means::Union{T, AbstractVector{T}} = T[]) where T <: Real
  data = vcat(points...)
  if length(impute_means) > 0
    data_filled = impute_points(data, impute_means)
  else
    data_filled = data
  end
  indices = Vector{UnitRange{Int}}()
  accum = 1
  for point_set in points
    n_points = size(point_set, 2)
    push!(indices, accum:(accum+n_points-1))
    accum += n_points
  end
  (tree = KDTree(data_filled, norm, leafsize=leaf_size),
   data = data_filled,
   indices = indices)
end

# convenience method that, in my opinion, should exist in `Base` as a means to avoid extra allocations
function collect!(x::AbstractArray, g::Base.Generator)
  for (i, v) in enumerate(g)
    x[i] = v
  end
end


using ProgressMeter

const default_max_n_points = 64_000 # an integer, lower bound on number of sparse entries per non-sparse entry

"""
  Both row and column points must reside in `points`, which corresponds exactly to the internal structure
  of the spatial tree.
"""
# `tolarance` signals that higher values are more relaxed, versus `threshold` that denotes the opposite
function eval_pairwise_cross_kernel(kernel::Kernel{T}, bandwidth::T, tree::Union{Nothing,NNTree},
    n_neighbors::Int, threshold::T, points::AbstractMatrix{T},
    row_vec_indices::AbstractVector{Int}, col_vec_indices::AbstractVector{Int},
    reference::Symbol; max_n_points::Int = default_max_n_points,
    impute_means::Union{T, AbstractVector{T}} = T[])::SparseMatrixCSC{T} where T
  row_indices, col_indices = OrderedSet(row_vec_indices), OrderedSet(col_vec_indices)
  row_index_map = Dict(r => i for (i, r) in enumerate(row_indices))
  n_dims, n_points = size(points)
  n_rows, n_cols = length(row_indices), length(col_indices)
  n_total = n_rows * n_cols
  #if max_n_points === nothing
  #  max_n_points = div(n_total, default_sparsity_ratio)
  #end
  #max_n_points::Int # does this serve at all as a hint to the compiler?
  if length(impute_means) > 0
    points_filled = impute_points(points, impute_means)
  else
    points_filled = points
  end
  col_points = @view points_filled[:, col_vec_indices]
  @assert !any( BroadcastArray(isnan, col_points) ) "somehow, imputed points are still NaN"
  #row_points = @view points[:, row_vec_indices]
  # possibility: individual range-based queries determined by the local kernel bandwidth? some multiplier on the effective Euclidean distance allowable? L1 >= L2 always
  if tree === nothing
    col_neighborhoods = [row_vec_indices for _ in col_vec_indices]
  else
    col_neighborhoods, _ = knn(tree, col_points, n_neighbors, false) # unsorted
  end
  #row_neighborhoods, _ = knn(tree, row_points, n_neighbors, false) # do from both sides to maintain symmetry
  all_displacements = zeros(T, n_dims, max_n_points) # serves as a staging area
  all_point_indices = fill((0, 0), 0)
  all_neighbors = zeros(Int, 0)
  all_evaluations = zeros(T, 0)
  batch_indices = zeros(Int, n_cols+1)
  batch_indices[1] = 1
  batch = 1
  offset = 0
  progress = ProgressUnknown()
  function rewind(batch_index)
    whole_range = 1:(batch_index - offset - 1) # an overengineered hot mess?
    orig_displacements = @view all_displacements[:, whole_range]
    point_indices = @view all_point_indices[whole_range .+ offset]
    if issparse(points)
      displacements = sparse(orig_displacements)
      # here I'm repurposing the `threshold` that was dedicated to kernel evaluations.
      # probably won't do much here... where would you find tiny albeit nonzero perturbations in high-dimensional data
      #droptol!(displacements, threshold) not worth the search, probably. `fkeeps!` makes it appear like no reallocation occurs, which is nice
      ratio = nnz(displacements) / length(displacements)
      next!(progress, showvalues=() -> [(:nonzero_percent, round(Int, 100ratio))])
      #@info "displacement batch had $(round(Int, 100ratio))% nonzero entries"
    else
      displacements = orig_displacements
      next!(progress)
    end
    evaluations = evaluate(kernel, bandwidth, displacements, one(T), point_indices)
    push!(all_evaluations, evaluations...)
    offset = batch_index - 1 # rewind to the beginning of `all_displacements`
  end
  for (col, hood) in zip(col_indices, col_neighborhoods)
    # thin out the column indices if they differ (i.e. we are doing an asymmetric diffusion)
    clean_hood = filter(index -> index in row_indices, hood) # Iterators.filter would be the lazy version
    #local_row_indices = (row_index_map[r] for r in clean_hood) this is cool and all but we don't actually need it (nor do we need `row_neighborhoods`)
    @debug "neighborhood" length(clean_hood)
    neighbors = @view points[:, clean_hood] # copy later instead to rearrange contiguously and maintain locality
    n_neighbors = length(clean_hood)
    point = @view points[:, col]
    batch_index = batch_indices[batch]
    batch_end = batch_index + n_neighbors-1
    if batch_end > max_n_points
      @debug "exceeded" max_n_points
      #max_n_points *= 2 max_n_points CODE FOR GROWING
      #new_all_displacements = zeros(T, n_dims, max_n_points)
      #new_all_point_indices = fill((0, 0), max_n_points)
      #new_all_neighbors = zeros(Int, max_n_points)
      #old_range = 1:size(all_displacements, 2)
      #new_all_displacements[:, old_range] = all_displacements
      #new_all_point_indices[old_range] = all_point_indices
      #new_all_neighbors[old_range] = all_neighbors
      #all_displacements = new_all_displacements
      #all_point_indices = new_all_point_indices
      #all_neighbors = new_all_neighbors
      rewind(batch_index)
    end
    batch_range = (batch_index:batch_end) .- offset
    # I mistakenly suspected that the batched assignment was causing spurious allocations
    @inbounds begin
      for (i, (b, r)) in zip(batch_range, clean_hood) |> enumerate
        # I think these can be implicitly sparse vectors coming from sparse matrices
        all_displacements[:, b] = neighbors[:, i] .- point
        if reference == :sym
          push!(all_point_indices, (r, col))
        elseif reference == :row # asymmetric
          push!(all_point_indices, (r, r))
        elseif reference == :col
          push!(all_point_indices, (col, col))
        else
          error("argument `reference` must be one of {:sym, :row, :col}")
        end
        push!(all_neighbors, r)
      end
    end
    #collect!(@view(all_point_indices[batch_range]), point_indices) <-- that was a generator. which is better? is it really wasteful to create generators in performance-critical loops? can't they be optimized out? the question is, are they?
    batch += 1
    batch_indices[batch] = batch_index + n_neighbors
  end
  rewind(batch_indices[end])
  finish!(progress)
  sparse_rows, sparse_cols, sparse_vals = zeros(Int, 0), zeros(Int, 0), zeros(T, 0)
  for (j, col) in enumerate(col_indices)
    batch_range = batch_indices[j]:(batch_indices[j+1] - 1)
    evaluations = @view all_evaluations[batch_range]
    neighborhood = @view all_neighbors[batch_range]
    for (r, e) in zip(neighborhood, evaluations)
      if e < threshold
        continue
      end
      local_row_index = row_index_map[r]
      push!(sparse_rows, local_row_index)
      push!(sparse_cols, j)
      push!(sparse_vals, e)
      # fill the other triangle, and combine using the `max` operation to have larger (nonzero) kernel take precedence when there is a conflict in the overlap
      # I guess, if I were guaranteeing to use the symmetric adaptive kernel, I could add one side each time and combine the overlaps with addition
      # this will always be balanced for the :sym cases, but maybe not the asymmetrical ones
      push!(sparse_cols, local_row_index)
      push!(sparse_rows, j)
      push!(sparse_vals, e)
    end
  end
  # pack it up. gobble gobble
  sparse(sparse_rows, sparse_cols, sparse_vals, n_rows, n_cols, max) # last one specifies the combinator function
end

function normalize_diffusion(affinities::SparseMatrixCSC{T}, alpha::T,
    )::Tuple{SparseMatrixCSC{T}, Matrix{T}} where T
  if alpha == 0
    return affinities, ones(T, size(affinities, 1), 1)
  end
  scale = sum(affinities, dims=2).^alpha # column indexing is faster
  diffusion = (affinities ./ scale) ./ scale' # careful not to expand the outer product of vectors
  return diffusion, scale' # `scale` comes out as a row vector. we can afford linearly-growing dense vectors---not quadratic
end

using KrylovKit # should support my domestic industry by trying out "functional-eigen.jl" !

function decompose_diffusion(diffusion::SparseMatrixCSC{T}, n_eig::Int;
    krylov_margin::Int = 10, tolerance::T = T(1e-5), threshold::T = T(0) # default tolerance is way too strict, and takes forever
    )::Tuple{Vector{T}, Matrix{T}, Matrix{T}} where T
  totals = sum(diffusion, dims=2).^0.5
  transitions = (diffusion ./ totals) ./ totals'
  # transitions are normalized in the symmetric fashion, but with overall magnitudes comparable to the row-stochastic rendition
  if threshold > 0
    transitions = copy(transitions)
    droptol!(transitions, threshold)
  end
  # `which=:LR` is needed because otherwise it sorts by largest magnitude, i.e. absolute value
  # there are more items in the tuple to unpack if I wish, pertaining to computational characteristics
  eigval, eigvec = eigsolve(transitions |> Hermitian, n_eig, :LR,
    issymmetric=true, krylovdim=n_eig+krylov_margin, tol=tolerance) # we need to establish a baseline for the krylov dimensions
  # ^ tell the solver that the effective subspace is of dimensionality equal to the number of vectors we seek
  # tying this to `n_eig` could possibly introduce problems when it is too small (for `n_eig` too small)
  eigvec = hcat(reverse(eigvec[1:n_eig]) ...) # could return more than requested
  eigval = reverse(eigval[1:n_eig])
  eigval, eigvec, totals
end

function embed_diffusion(diffusion::SparseMatrixCSC{T}, n_eig::Int
    )::Tuple{Vector{T}, Matrix{T}, Vector{T}} where T
  eigval, eigvec, totals = decompose_diffusion(diffusion, n_eig)
  transition_eigvec = eigvec ./ totals
  coordinates = transition_eigvec .* eigval'
  density = dropdims(totals.^2, dims=2)
  return eigval, coordinates, density
end

function estimate_implicit_embedding(kernel::Kernel{T}, points::AbstractMatrix{T};
    bandwidth::T, alpha::T, n_manifold_dims::Int, leaf_size::Int, n_neighbors::Int, threshold::T) where T
  @assert 0 <= alpha <= 1
  @assert bandwidth > 0
  structure = build_tree(points; leaf_size, norm=Cityblock())
  affinities = eval_pairwise_cross_kernel(kernel, bandwidth, structure.tree,
    n_neighbors, threshold, points, structure.indices[1], structure.indices[1], :sym)
  diffusion, _ = normalize_diffusion(affinities, alpha) # OCaml style (i.e. with currying) would be to have alpha as the first argument
  eigens, coords, _ = embed_diffusion(diffusion, n_manifold_dims+1)
  (eigenvalues = eigens[1:end-1],
   coordinates = coords[:, 1:end-1])
end

# best to repeat code here from explicit-diffusion.jl and have two separate, flexible paths at the cost of a modicum of redundancy

using Conda, PyCall
# I've been having lots of trouble with pandas DataFrames' interoperability with Parquet as an intermediate format
function read_scipy_sparse_matrix(filename::String)::SparseMatrixCSC
  scipy = pyimport_conda("scipy.sparse", "scipy")
  matrix = scipy.load_npz(filename).tocoo() # could already be COO (and ideally would)
  sparse(matrix.row .+ 1, matrix.col .+ 1, matrix.data)
end
