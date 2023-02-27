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

# implement landmarks exactly as prescribed in the 2019 Nature Biotechnology paper,

# power the eigenvalues, watch all but the first vanish..

# after some embedding step (MDS for vanilla/pure PHATE,) landmark points are basically
# interpolated in a linear fashion to the comprehensive set via the transition matrix

using Clustering
using LinearAlgebra
using Statistics
using KrylovKit
using SparseArrays


"""
  We do not center by the mean here, as we assume the matrix is so sparse that the mean is
  effectively zero. This enables an efficient procedure for sparse matrices.
"""
function reduce_to_sparse_principal_components(data::SparseMatrixCSC{T, Int},
    n_dims::Int, tolerance::T, krylov_margin::Int = 10) where T
  n_points = size(data, 2)
  # increases the number of non-sparse entries typically by an order of magnitude
  product = data * data' / n_points
  droptol!(product, tolerance)
  # Hermitian simply forces symmetry so that the algorithm may converge harmoniously.
  # does the added overhead slow down the sparse multiplication?
  eigval, eigvec = eigsolve(product |> Hermitian, n_dims, :LR,
    issymmetric=true, krylovdim=n_dims+krylov_margin) # see implicit-diffusion.jl for justification
  projection = hcat(eigvec[1:n_dims] ...)'
  reduced = projection * data
  eigenvalues = eigval[1:n_dims]
  (; reduced, projection, eigenvalues)
end

function extract_transitions(diffusion::SparseMatrixCSC{T, Int},
    scale::AbstractMatrix{T}) where T
  n_points = size(diffusion, 1)
  @assert size(diffusion, 1) == size(diffusion, 2)
  density = scale.^2 # `scale` was square-rooted sum, initially
  transitions = diffusion ./ density # transpose all of this a priori?
  transitions, density # this boiled down to a pretty thin method overall
end

"""
  Receives the diffusion matrix post normalization, but not made row-stochastic.
"""
function extract_landmark_transitions(diffusion::SparseMatrixCSC{T, Int},
    eigvec::AbstractMatrix{T}, scale::AbstractMatrix{T}, n_clusters::Int;
    threshold::T = T(0)) where T
  n_points = size(diffusion, 1)
  @assert size(diffusion, 1) == size(diffusion, 2)
  # at this point we don't throw away the largest eigenvector, which corresponds to the stationary distribution
  transitions, density = extract_transitions(diffusion, scale)
  thin_transitions = copy(transitions)
  droptol!(thin_transitions, threshold) # has already been thresholded via `affinities,` so reserve this only for projecting
  # v  I IDENTIFIED THIS AS THE FINAL REMAINING SLOW LINE
  reduced = thin_transitions * eigvec # we will perform a sort of spectral clustering with eigenvectors serving as a linear basis
  n_dims = size(eigvec, 2)
  @assert size(reduced) == (n_points, n_dims)
  clusters = kmeans(reduced', n_clusters, init=:kmcen) # `kmeans` accepts matrix as (n_dims x n_samples). these spectral coordinates come as (n_samples x n_dims)
  labels = assignments(clusters)
  @assert minimum(labels) == 1 && maximum(labels) == n_clusters
  @assert length(labels|>unique) == n_clusters # each label should have at least one assignment
  forward = zeros(T, n_points, n_clusters) # from point to cluster
  sources = zeros(T, n_clusters, n_points)
  backward = zeros(T, n_clusters, n_points)
  #for c in 1:n_clusters
    #mask = (labels .== c)
    # THE SLOWEST PART HERE IS PRECISELY `transitions[mask, :]` due to indexing the rows of every column in a funky way
    #forward[:, c] = dropdims(sum(
    #    @view(transitions[:, mask]),
    #  dims=2), dims=2)
    # the graphtools Github project does not include the below in their landmark code, even though the paper mentions it
    #sources[c, mask] = density[mask, 1] ./ sum(density[mask, 1]) # Q(j, xi) = density[xi] / sum_(zeta in cluster j) density[zeta]
    #backward[c, :] = dropdims(sum(
    #    @view(transitions[mask, :]) .* @view(sources[c, mask]),
    #  dims=1), dims=1) REPLACED BY THE BELOW. phasing out the uncouth loop
  #end
  for (p, c) in enumerate(labels)
    sources[c, p] = density[p, 1]
  end
  source_sum = sum(sources, dims=2)
  @assert all( source_sum .> 0 ) "transition sums should be positive: $source_sum"
  sources ./= source_sum
  # efficiently traverse the sparse matrix, one column at a time
  rows, vals = rowvals(transitions), nonzeros(transitions)
  for col in 1:n_points
    col_c = labels[col]
    for row in nzrange(transitions, col)
      p = rows[row]
      row_c = labels[p] # emulate `(p, c) in enumerate(labels)`
      transition = vals[row] # transitions[p, col]
      forward[p, col_c] += transition
      backward[row_c, col] += transition * sources[row_c, p]
    end
  end
  @debug "Markov sums" sum(sources, dims=2), sum(forward, dims=2), sum(backward, dims=2)
  # the `droptol!`, which I no longer do, is responsible for having the long sums tally slightly below unit
  (; forward, backward, labels)
end

"""
  Accepts a row-stochastic Markov transition matrix.
""" # I appreciate this distance metric. It feels less ad-hoc than the so-called "diffusion distance" that is actually an L^2-norm of the probabilities.
function compute_potential_distances(transitions::AbstractMatrix{T},
    gamma::T = T(1))::Matrix{T} where T
  n_points = size(transitions, 1)
  @assert size(transitions, 1) == size(transitions, 2)
  if gamma == 1
    f = (p::T) -> log(max(p, floatmin(T))) # this syntax as opposed to f(p) = ... forces the function to be a local variable, so the compiler doesn't complain about redefining it
  elseif gamma == -1
    f = (p::T) -> p
  else
    f = (p::T) -> 2/(1-gamma) * p^((1-gamma)/2)
  end
  distances = zeros(T, n_points, n_points)
  log_transitions = f.(transitions)
  # partition the `i, j` (or simply `i`) space and parallelize. beware of data races when writing to the completely unprotected `distances[j, i]`. parallel reads are safe
  for k in 1:n_points # this is beginning to look spaghettified like typical C
    Threads.@threads for i in 1:n_points # this somehow makes it like multiple orders of magnitude faster. why? cache coherence per core?
      for j in 1:(i-1) # need to put it on a separate line this time
        difference = log_transitions[i, k] - log_transitions[j, k]
        distances[j, i] += difference ^ 2 # this per se might not even be that much faster than the below commented-out code
      end
    end
  end
  for i in 1:n_points, j in 1:(i-1)
    # here accessing both (i,j) and (j,i) in some manner is unavoidable
    distances[i, j] = distances[j, i]
  end
  distances .|> sqrt
  #[ sum((
  #    f.(@view transitions[i, :]) - f.(@view transitions[j, :])
  #  ) .^ 2) .^ 0.5
  #  for i in 1:n_points, j in 1:n_points ]
end

using PyCall, Conda

function perform_classic_mds(distances::AbstractMatrix, n_manifold_dims::Int)
  squared = distances .^ 2
  semi_centered = distances .- mean(distances, dims=1)
  centered = semi_centered .- mean(semi_centered, dims=2)
  embedding = reduce_to_principal_components(centered, n_manifold_dims)
  embedding.reduced
end

function perform_metric_mds(distances::AbstractMatrix{T},
    n_manifold_dims::Int)::AbstractMatrix{T} where T
  # classic MDS serves as initialization for a warm start
  # see https://github.com/KrishnaswamyLab/PHATE/blob/master/Python/phate/mds.py
  classic_result = perform_classic_mds(distances, n_manifold_dims)
  # automatically calls `Conda.add` if package is not installed and user runs a Julia-local conda environment
  skl_manifold = pyimport_conda("sklearn.manifold", "scikit-learn")
  # note that here, as well as in graphtools, the stress function does not weigh disparities by true dissimilarity as the paper puts forth
  # wait, why is `metric=false` slower? is it better? the scipy code makes it look like it applies some corrective procedure to the `distances`
  embedding, stress, n_iter = skl_manifold.smacof(distances, metric=true,
    n_components=n_manifold_dims, return_n_iter=true, n_init=1, init=classic_result')
  convert.(T, embedding)
end

function fit_line(x::AbstractVector{T}, y::AbstractVector{T}) where T <: Real
  x_bar, y_bar = mean(x), mean(y)
  numerator = sum((x .- x_bar) .* (y .- y_bar))
  denominator = sum((x .- x_bar).^2)
  slope = numerator / denominator # how to represent an infinnite slope? don't: filter for NaNs later
  intercept = y_bar - slope*x_bar
  fit = intercept .+ slope.*x
  error = sum((fit .- y).^2)
  (; slope, intercept, error)
end

# very simple method, which I believe is the same one harnessed by and for PHATE
function find_knee(x::AbstractVector{T}, y::AbstractVector{T})::T where T <: Real
  @assert length(x) == length(y)
  n_points = length(x)
  errors = zeros(T, n_points)
  for (i, pivot) in enumerate(x)
    # do not exclude the pivot for symmetry. we need to sum over the same set of items each time for commensurable error tallies
    left_mask = (x .<= pivot)
    right_mask = .!left_mask
    left_fit = @views fit_line(x[left_mask], y[left_mask])
    right_fit = @views fit_line(x[right_mask], y[right_mask])
    total_error = left_fit.error + right_fit.error
    @debug pivot left_fit right_fit
    errors[i] = total_error
  end
  masked_errors = ifelse.(isfinite.(errors), errors, Inf)
  # possibly disallow lines of length 1 and 2 as there are 2 degrees of freedom
  best_index = argmin(masked_errors[2:end-1]) + 1
  x[best_index]
end

function estimate_von_neumann_entropy(eigval::AbstractVector{T})::T where T <: Real
  positive_eigval = eigval[eigval .> 0]
  normalizer = sum(positive_eigval)
  probabilities = positive_eigval / normalizer
  pointwise_entropies = -probabilities .* log.(probabilities)
  sum(pointwise_entropies)
end

function optimize_von_neumann_entropy(eigval::Vector{T},
    powers::AbstractRange{T})::T where T # <:Real is implied
  n_powers = length(powers)
  x, y = zeros(T, n_powers), zeros(T, n_powers)
  for (i, t) in enumerate(powers)
    step_eigval = max.(eigval, 0) .^ t
    entropy = estimate_von_neumann_entropy(step_eigval)
    x[i] = t
    y[i] = entropy
  end
  knee = find_knee(x, y)
  #convert(Int, knee) # no InexactErrors here
end

# why not @NamedTuple at this point? tuples are covariant in their parameters...
struct LandmarkFit{T <: Real}
  coordinates :: Matrix{T}
  landmark_distances :: Matrix{T}
  landmarks :: Matrix{T}
  forward_projection :: Matrix{T}
  backward_projection :: Matrix{T}
  eigenvalues :: Vector{T}
  entropy :: T
end

using NaNMath

"""
  Do not tempt phate!

  If `n_landmarks <= 0`, produce a one-to-one correspondence between landmarks and original points.
"""
function leave_it_to_phate(kernel::Kernel{T}, points::AbstractMatrix{T}; bandwidth::T, n_landmarks::Int,
    n_bases::Int, n_manifold_dims::Int, tree_leaf_size::Int, n_kernel_neighbors::Int,
    cuda_devices::Vector{Int}=Int[], sparsity_threshold::T, alpha::T, eig_tolerance::T = T(1e-5),
    n_steps::T = T(-1), step_powers::AbstractRange{T} = 1:T(100) )::LandmarkFit{T} where T <: Real
  n_dims, n_points = size(points)
  #if issparse(points)
  #  imputation = zeros(T, n_dims)
  imputation = dropdims(mapslices(NaNMath.median, points, dims=2), dims=2)
  if tree_leaf_size > 0
    structure = build_tree(points, leaf_size=tree_leaf_size,
      norm=Cityblock(), impute_means=imputation)
    tree = structure.tree
  else
    # tree is sometimes just not worth it...
    tree = nothing
  end
  kernel_time = @elapsed if length(cuda_devices) == 0
    affinities = eval_pairwise_cross_kernel(
      kernel, bandwidth, tree, n_kernel_neighbors, sparsity_threshold,
      points, 1:n_points, 1:n_points, :sym, impute_means=imputation)
  else
    @assert CUDA.functional()
    dense_affinities = par_evaluate_pairwise(
      kernel, bandwidth, points, cuda_devices)
    dense_affinities .= ifelse.(
        dense_affinities .> sparsity_threshold, dense_affinities, T(0) )
    affinities = sparse(dense_affinities)
  end
  @debug "NaNs out of nonzero entries" sum(isnan.(affinities)) sum(affinities .> 0)
  @info "evaluated kernels in $(round(Int, kernel_time)) seconds"
  @assert !any( isnan.(affinities) ) "no kernelized affinities can be NaN"
  diffusion, _ = normalize_diffusion(affinities, alpha) # PHATE does not do this per se
  # recycle/reuse expensive results
  eigen_time = @elapsed begin
    eigval, eigvec, scale = decompose_diffusion(diffusion, n_bases,
      tolerance=eig_tolerance, threshold=sparsity_threshold) # from implicit-diffusion.jl
  end
  @info "eigen-decomposed the large sparse matrix in $(round(Int, eigen_time)) seconds"
  one_step_entropy = estimate_von_neumann_entropy(eigval)
  # it's okay that n_steps is actually a float because
  # all matrix power methods check for `isinteger` first
  if n_steps == -1
    n_steps = optimize_von_neumann_entropy(eigval, step_powers)
    @info "optimal no. of steps looks to be $n_steps"
  end
  if n_landmarks <= 0
    landmark_time = @elapsed begin
      all_transitions, _ = extract_transitions(diffusion, scale) # ( ... ;)
    end
    # the landmark method takes care of renormalization
    transitions = ( forward=Matrix{T}(I, n_points, n_points),
      backward=convert(Matrix, all_transitions), labels=1:n_points ) # ensure type stability, for the most part
  else
    landmark_time = @elapsed transitions = extract_landmark_transitions(
      diffusion, eigvec, scale, n_landmarks, threshold=sparsity_threshold)
  end
  compact_transitions = transitions.backward * transitions.forward
  @assert(
    isapprox.(sum(compact_transitions, dims=2), 1) |> all,
    "$(sum(compact_transitions, dims=2)) should be unit" )
  # if there are negative eigenvalues, then taking a real power returns
  # an imaginary matrix (imaginary parts should be tiny though)
  diffused_transitions = compact_transitions^n_steps |> real .|> T
  diffused_transitions ./= sum(diffused_transitions, dims=2) # corrective post-hoc renormalization
  @info "found landmarks in $(round(Int, landmark_time)) seconds" diffused_transitions
  distances = compute_potential_distances(diffused_transitions)
  mds_time = @elapsed begin
    compact_coordinates = perform_metric_mds(distances, n_manifold_dims)
    # just always realign with PCA
    # (as MDS initializes that way but does not guarantee it in the end result)
    principal = reduce_to_principal_components(compact_coordinates', n_manifold_dims)
    compact_coordinates = permutedims(principal.reduced)
  end
  @info "multidimensionally scaled in $(round(Int, mds_time)) seconds"
  coordinates = transitions.forward * compact_coordinates # linear re-interpolation to back out full/complete coordinates
  #(; coordinates, distances, landmarks=compact_coordinates,
  #  forward_projection=transitions.forward, backward_projection=transitions.backward)
  LandmarkFit(coordinates, distances, compact_coordinates,
    transitions.forward, transitions.backward, eigval, one_step_entropy)
end # the only thing left is the branch-identification portion


using Statistics

"""
  Estimate the within-cluster variance of some real-valued attribute.
  If clustering is based on the potential-distance matrix, either
  figure out how to interpolate the inter-landmark distance to the
  entire set of points, or centralize the attributes to approximate
  landmark values.
  If clusters can reduce variance, that is good. It hints that they
  can separate points with differing attributes. Like ANOVA?
  We also return the means, in case one wishes to correlate an
  attribute with some grouping mechanism, like a label to be predicted.
""" # could also estimate categorical entropy for discrete attributes. variance here upper-bounds the differential entropy
function estimate_cluster_attribute_variances(
    clusters::Vector{Int}, attributes::Vector{T}) where T <: Real
  n_clusters = maximum(clusters)
  @assert minimum(clusters) == 1
  n_points = length(attributes)
  @assert n_points == length(clusters)
  means = zeros(T, n_clusters)
  variances = zeros(T, n_clusters)
  agglomerate = zero(T)
  for cluster in 1:n_clusters
    # pay less heed to efficiency here
    mask = (clusters .== cluster)
    cluster_attributes = @view attributes[mask]
    cluster_mean = mean(cluster_attributes)
    cluster_residuals = (cluster_attributes .- cluster_mean) .^ 2
    means[cluster] = cluster_mean
    variances[cluster] = mean(cluster_residuals)
    agglomerate += sum(cluster_residuals)
  end
  agglomerate_score = sqrt(agglomerate / n_points)
  (; score=agglomerate_score, means, variances ) # (x=y; z, w) ??
end

"""
  Finds a cluster hierarchy by Ward linkages and estimates attribute
  variances on a specified cut. At this point I am taking a few
  liberties and deviating from the original PHATE analysis.
"""
function estimate_landmark_cluster_attribute_variances(
    fit::LandmarkFit{T}, n_clusters::Int, attributes::Vector{T}) where T
  landmark_attributes = fit.backward_projection * attributes
  cluster_hierarchy = hclust(fit.landmark_distances, linkage=:ward)
  clusters = cutree(cluster_hierarchy, k=n_clusters)
  estimate_cluster_attribute_variances(clusters, landmark_attributes)
end

# thought: somehow sort clusters topologically and by average cell age?

using PyCall

"""
  We need attributes to be densely packed integers (i.e. 1, 2, 3, ...).
"""
function estimate_landmark_cluster_attribute_homogeneity(
    fit::LandmarkFit{T}, n_clusters::Int, attributes::Vector{Int})::T where T
  #possible_labels = unique(attributes)
  soft_attributes = convert.(T, attributes)
  soft_landmark_attributes = fit.backward_projection * soft_attributes
  landmark_attributes = round.(Int, soft_landmark_attributes)
  cluster_hierarchy = hclust(fit.landmark_distances, linkage=:ward)
  clusters = cutree(cluster_hierarchy, k=n_clusters)
  metrics = pyimport_conda("sklearn.metrics", "scikit-learn")
  metrics.homogeneity_score(landmark_attributes, clusters)
end
