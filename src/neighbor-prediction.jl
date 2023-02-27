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

# could dispatch on `extend` by making it a Val{Bool}.
# it would be nice if you could specify parameters for function types,
# meaning that it needs to contain a method satisfying that signature

# two options here: either delegate all training and predicting to a
# single master closure `train_and_predict`, or accept two functions
# `train` and `predict` that take in `predictor_params...` and are
# fed all the arguments necessary to act on their own without external
# state. this latter approach would make it simpler to specify and call.

using SpecialFunctions
using Distances: Euclidean
using StatsBase
using StaticArrays
using Flux, Zygote # ChainRulesCore.rrule is doper

# multivariate Beta function
log_multi_beta(v) = sum(loggamma, v) - loggamma(v |> sum)

# this is the implicit deconstructing notation for lambdas/closures
function log_dirichlet(x, a)
  @assert all(x .>= 0) && sum(x) == 1 "Dirichlet support is on the unit simplex"
  sum(((xi, ai),) -> (ai-1)log(xi), zip(x, a)) - log_multi_beta(a)
end

# the Dirichlet-multinomial distribution, also known as the multivariate Polya distribution,
# is the distribution that comes out of marginalizing a multinomial distribution over the
# support of its Dirichlet prior. log_dirichlet of a vector like [0,0,0,1,0] is -Inf, but
# the log-probability of its marginalized multinomials reduces to a very simple form.
# keep in mind that beta(x,1) = 1/x, so logbeta(a0, 1) - logbeta(ai, 1) = log(ai) - log(a0),
# the freaking categorical distribution!! that's rather magnificent.
# having every prior "pseudocount" alpha_i := 1 corresponds to the multivariate version
# of Laplace's rule of succession.. so cool!!! it's all connected.
# use it to maintain finite individual log-likelihoods, to be summed up
#log_single_polya_draw(index, alphas) = ...

"""
  Operates on Euclidean embeddings. Labels can be arbitrary integers.
  -- Negative labels are treated as missing, and ignored.
  `embedding` should be (n_dims x n_points).
  We recommend setting `prior_count := 1.0` if there is no other prior indication.
""" #`countmap` (`counts` for integral indices) is so useful by the way
function score_neighborhood_classifier(embedding::AbstractMatrix{T}, labels::AbstractVector{Int};
    n_neighbors::Int, leaf_size::Int=0, compare_baseline::Bool=false, temperature::T=T(0), # so much configurability now...
    prior_count::T=T(0), global_prior::Bool=false, verbose::Bool=false)::Vector{T} where T <: Real
  if compare_baseline
    score = score_neighborhood_classifier(embedding, labels, compare_baseline=false;
      n_neighbors, prior_count, global_prior, leaf_size, verbose)
    # prior_count := 1 for safety in case there are singular instances of particular labels
    # for when I did `n_neighbors=n_baseline_neighbors, prior_count=T(1)`
    # rather, I will set neighbrs to zero and invoke a global prior, something that is MUCH faster
    baseline = score_neighborhood_classifier(embedding, labels,
      n_neighbors=0, global_prior=true, compare_baseline=false; leaf_size, verbose)
    return score - baseline
  end
  n_categories = 0; posterior_total = 0; label_counts = Dict() # assign to this scope
  Zygote.ignore() do
    n_categories = sum(unique(labels) .>= 0)
    if global_prior
      posterior_total = sum(>=(0), labels) + n_neighbors
      label_counts = countmap(labels)
    else
      posterior_total = prior_count*n_categories + n_neighbors # posterior alpha_0
    end
  end
  n_dims, n_points = size(embedding)
  @assert n_points == length(labels)
  indicators = construct_neighborhood_indicators(embedding;
    temperature, leaf_size, n_neighbors, verbose)
  progress = Progress(length(labels),
    desc="Scoring... ", enabled=verbose)
  # indicators can be real OR integral. flexible implementation below. BUT HENCEFORTH TYPE STABILITY IS ENFORCED
  map(labels|>enumerate) do (point_index, label)
    # don't want to carry around whole SVector{n_points}s
    neighborhood = @view indicators[:, point_index]
    next!(progress)
    if label < 0
      return T(NaN)
    end
    filtered_neighbors = Iterators.filter(
        zip(labels, neighborhood)) do (neighbor_label, neighbor_weight)
      neighbor_label == label
    end
    n_hits = sum(filtered_neighbors) do (neighbor_label, neighbor_weight)
      neighbor_weight
    end
    this_prior_count = global_prior ? label_counts[label] : prior_count
    log_likelihood = log((n_hits + this_prior_count) / posterior_total) |> T
  end
end

# use the mean as a test statistic and draw from an empirical null distribution
# a permutation test?
function test_neighborhood_classifier(embedding::AbstractMatrix{T}, labels::AbstractVector{Int};
    n_neighbors::Int, n_null_draws::Int, leaf_size::Int=0, temperature::T=T(0),
    prior_count::T=T(0), global_prior::Bool=false, verbose::Bool=false) where T <: Real
  shuffled_labels = copy(labels)
  progress = Progress(n_null_draws,
    desc="Drawing from the null... ", enabled=verbose)
  null = map(1:n_null_draws) do i
    shuffle!(shuffled_labels)
    draw = score_neighborhood_classifier(embedding, shuffled_labels;
      n_neighbors, leaf_size, temperature, prior_count, global_prior)
    next!(progress)
    NaNMath.mean(draw)
  end
  alternative = score_neighborhood_classifier(embedding, labels;
    n_neighbors, leaf_size, temperature, prior_count, global_prior)
  probability = alternative |> NaNMath.mean |> ecdf(null)
  1 - probability, null
end


function score_neighborhood_spread(embedding::AbstractMatrix{T}, values::AbstractMatrix{T};
    n_neighbors::Int, leaf_size::Int=0, temperature::T=T(0), norm_power::T=T(2),
    downselect::Function=(i,j)->true, verbose::Bool=false)::Matrix{T} where T <: Real
  # you could in principle incorporate a baseline to adjust for heterogeneous spreads across the value distribution
  n_dims, n_points = size(embedding)
  n_vars, n_val_points = size(values)
  @assert n_points == n_val_points
  values = permutedims(values) # for speed later on
  indicators = construct_neighborhood_indicators(embedding;
    temperature, leaf_size, n_neighbors, verbose)
  result = zeros(T, n_points, n_vars)
  progress = Progress(n_vars * n_points,
    desc="Scoring... ", enabled=verbose)
  # wouldn't it be amazing if `mapreduce(hcat, 1:n_vars) do var_index`
  # automagically knew to instantiate one large matrix of a certain
  # size and fill it all in column by column?
  @threads for (res_index, var_index) in axes(values, 2) |> enumerate |> collect
    # interoperable with arrays of any indexing mechanism? is this the most correct way?
    # at some point this general accommodation becomes a little too much.
    value_slice = @view values[:, var_index]
    for (point_index, value) in enumerate(value_slice)
      neighborhood = @view indicators[:, point_index]
      # if you choose to downselect based on index, you may get fewer than `n_neighbors` neighbors
      selection = filter(j -> downselect(point_index, j), 1:n_points) # avoid extra allocation when I can?
      neighborhood = @view neighborhood[selection]
      neighbor_value_slice = @view value_slice[selection]
      next!(progress)
      residual = sum(zip(neighbor_value_slice, neighborhood)
          ) do (neighbor_value, neighbor_weight)
        neighbor_weight * abs(value - neighbor_value) ^ norm_power
      end
      spread = (residual / sum(neighborhood)) ^ (1/norm_power)
      result[point_index, res_index] = spread
    end
  end
  result |> permutedims
end

# use the mean as a test statistic and draw from an empirical null distribution
# a permutation test?
function test_neighborhood_spread(embedding::AbstractMatrix{T}, values::AbstractVector{T};
    n_neighbors::Int, n_null_draws::Int, leaf_size::Int=0, temperature::T=T(0),
    norm_power::T=T(2), verbose::Bool=false) where T <: Real
  # turn vector into matrix. for now, no support for concurrent tests
  shuffled_values = permutedims(values)
  progress = Progress(n_null_draws,
    desc="Drawing from the null... ", enabled=verbose)
  null = map(1:n_null_draws) do i
    shuffle!(shuffled_values)
    draw = score_neighborhood_spread(embedding, shuffled_values;
      n_neighbors, leaf_size, temperature, norm_power)
    next!(progress)
    NaNMath.mean(draw)
  end
  alternative = score_neighborhood_spread(embedding, values;
    n_neighbors, leaf_size, temperature, norm_power)
  probability = alternative |> NaNMath.mean |> ecdf(null)
  probability, null
end


function construct_neighborhood_indicators(embedding::AbstractMatrix{T};
    temperature::T, leaf_size::Int, n_neighbors::Int, verbose::Bool)::Matrix{T} where T
  n_dims, n_points = size(embedding)
  if temperature == 0 # NaN as more suitable flag?
    @assert leaf_size > 0
    tree_time = @elapsed begin
      tree = KDTree(embedding, Euclidean(),
        leafsize=leaf_size, reorder=false) # last setting for performance?
      indices, distances = knn(tree, embedding, n_neighbors+1, true)
    end
    if verbose
      @info "Constructed and searched the spatial tree in $(round(Int, tree_time)) seconds."
    end
    progress = Progress(length(indices),
      desc="Sifting... ", enabled=verbose)
    indicators = zeros(T, n_points, n_points) # type stability across branches
    for (col, neighborhood) in enumerate(indices)
      for neighbor in @view neighborhood[2:end]
        indicators[neighbor, col] += 1
      end
      next!(progress)
    end
  else
    @assert temperature > 0 # softened, differentiable version
    distances = pairwise(Euclidean(), embedding, dims=2)
    weights = .-distances - Inf*I # remove diagonal (self-proximity) part
    top = zeros(T, n_points, n_points)
    for k in 1:n_neighbors
      # Zygote-differentiable code (without a custom backend) favors NumPy-style
      # batched operations over individuals in-place mutations
      soft_prox = exp.( weights ./ temperature )
      softmax = soft_prox ./ sum(soft_prox, dims=1)
      top += softmax # do not mutate array
      # see the update rule in "Reparameterizable Subset Sampling via Continuous Relaxations," 2021.
      weights += log.(1 .- softmax)
    end
    indicators = top
  end
  indicators
end
