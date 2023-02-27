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

"""
  Computes the earth mover's distance on empirical distributions defined in sparse matrices.
  First matrix should contain gene expression levels as columns and cells as rows.

  Weights allow you to control the units of mass that each point contributes to the empirical
  distribution function. They are automatically rescaled.
    For instance, if two points supposedly belong to the same cell, then you can halve their
    weights to ensure that cell does not have an outsize impact on the EMDs.

  The Dirichlet scale represents alpha_0 := sum_i alpha_i of the prior for Bayesian bootstrapping.
  No actual resampling occurs; the drawing from a multinomial (like in the multivariate Polya
  distribution) is implicit through weights. For inexact resample sizes, we could have drawn from
  Poisson variables.

  Returns a matrix of dimensionality (n_clusters x n_genes).
""" # see https://github.com/scipy/scipy/blob/v1.6.1/scipy/stats/stats.py#L7701-L7775
function cluster_to_background_emd(
    data::SparseMatrixCSC{T, Int}, cluster_assignments::Vector{Int},
    weights::Vector{T} = T[]; dirichlet_scale::T = T(Inf))::Matrix{T} where T <: Real
  n_cells, n_genes = size(data) # metaphorical language that could actually refer to anything
  if length(weights) == 0
    weights = ones(T, n_cells)
  else
    weights = weights * (n_cells / sum(weights)) # `n_cells::Int` is implicitly cast with division. COPY RATHER THAN MUTATE
  end
  if isfinite(dirichlet_scale)
    # make sure this is technically correct.
    #  dirichlet_scale=1 -> sum up to n_cells, not unit,
    #  to represent having observed every cell once a priori
    prior = Dirichlet(weights * dirichlet_scale)
    observation_weights = rand(prior, n_cells)
    weights = sum(observation_weights, dims=2)[:, 1]
  end
  @assert length(cluster_assignments) == n_cells
  n_clusters = maximum(cluster_assignments)
  @assert minimum(cluster_assignments) == 1
  rows, vals = rowvals(data), nonzeros(data)
  distances = zeros(T, n_clusters, n_genes)
  background_size = sum(weights) # background "cluster" has all cells
  cluster_sizes = [
    weights[cluster_assignments .== cluster] |> sum
    for cluster in 1:n_clusters ]
  Threads.@threads for gene in 1:n_genes
    background_sample = zeros(T, 0)
    background_cells = zeros(Int, 0)
    cluster_samples = [ zeros(T, 0) for c in 1:n_clusters ]
    cluster_cells = [ zeros(Int, 0) for c in 1:n_clusters ]
    for r in nzrange(data, gene)
      cell = rows[r]
      expression = vals[r]
      cluster = cluster_assignments[cell]
      push!(background_sample, expression)
      push!(background_cells, cell) # to keep track of each expression's cell ID
      push!(cluster_samples[cluster], expression)
      push!(cluster_cells[cluster], cell)
    end
    if background_cells |> isempty
      # we could probably shortcut this path dramatically, but it ain't worth it
      background_absent_weight = sum(weights) # also `n_cells`
    else
      background_absent_weight = zero(T)
      i = 1
      for (cell, weight) in enumerate(weights)
        while ( i < length(background_cells)
            && background_cells[i] < cell )
          i += 1
        end
        if background_cells[i] != cell # missing
          background_absent_weight += weight
        end
      end
    end
    # implicitly, we must pad each sample with zeros until it reaches size `n_cells`
    # it is not the same to look at it as distributions of nonzero expressions with varying sizes
    sort!(background_sample)
    for cluster in 1:n_clusters
      cluster_sample = cluster_samples[cluster]
      this_cluster_cells = cluster_cells[cluster] # English can be clumsy sometimes, leading to inherent inconsistencies in naming schemes based on it
      cluster_size = cluster_sizes[cluster]
      sort!(cluster_sample)
      background_sample_size = length(background_sample)
      cluster_sample_size = length(cluster_sample) # number of nonzero entries for this cluster and this gene
      total = zero(T) # HEREIN gene expression are assumed to be non-negative
      if this_cluster_cells |> isempty
        cluster_absent_weight = cluster_size
      else
        cluster_absent_weight = zero(T)
        j = 1
        for (cell, weight) in enumerate(weights)
          if cluster_assignments[cell] != cluster
            continue
          end
          while ( j < length(this_cluster_cells)
              && this_cluster_cells[j] < cell )
            j += 1
          end
          if this_cluster_cells[j] != cell # missing, `>` if not at the end of the cluster's cells
            cluster_absent_weight += weight
          end
        end
      end
      # before the introduction of weights, the below used to be cluster_sample_size/cluster_size - background_sample_size/background_size
      # (n_cells - background_sample_size) - (n_cells - cluster_sample_size); start at effectively bin zero
      distance = ( background_absent_weight/background_size
        - cluster_absent_weight/cluster_size )
      i, j = 1, 1 # referring to background and cluster one-step-ahead indices, respectively
      t = zero(T)
      background_unit, cluster_unit = 1/background_size, 1/cluster_size
      while i <= background_sample_size || j <= cluster_sample_size
        update::T = 0
        # in reality, there is no way we reach the end of the cluster sample before the background sample because background subsumes cluster. be safe anyways
        if j > cluster_sample_size || background_sample[i] < cluster_sample[j]
          background_weight = weights[ background_cells[i] ]
          background_update = background_unit * background_weight
          interval = background_sample[i] - t
          t = background_sample[i]
          i += 1
          update = background_update # is this variable type-stable? would it help to ensure it explicitly?
        elseif background_sample[i] == cluster_sample[j] # common overlapping point; (do not) skip forthwith
          background_weight = weights[ background_cells[i] ]
          cluster_weight = weights[ this_cluster_cells[j] ]
          background_update = background_unit * background_weight
          cluster_update = cluster_unit * cluster_weight
          interval = background_sample[i] - t
          t = background_sample[i]
          i += 1
          j += 1
          update = background_update - cluster_update
        else # this will never get called since all cluster points also live in the background. but if I ever change that, the code will be here!
          cluster_weight = weights[ this_cluster_cells[j] ]
          cluster_update = cluster_unit * cluster_weight
          interval = cluster_sample[j] - t
          t = cluster_sample[j]
          j += 1
          update = -cluster_update
        end
        total += abs(distance) * interval
        distance += update # one more unit moved, regardless of whether it came from the background or the cluster
      end
      distances[cluster, gene] = total
    end
  end
  distances
end

using StatsBase

"""
  A permutation test for the two-sample EMD could be done by simulating the
  statistic under the null hypothesis, wherein points are interchanged randomly
  between the two samples. To generalize, we introduce a permutation test here
  that shuffles the `cluster_assignments` such that cluster cardinality is
  maintained. The `weights` vector remains coupled with the `data`.

  No option for Dirichlet draws here because that is a separate modus operandi.

  Returns a p-value.
"""
function test_cluster_to_background_emd(
    data::SparseMatrixCSC{T, Int}, cluster_assignments::Vector{Int},
    weights::Vector{T} = T[]; n_null_draws::Int )::Matrix{T} where T <: Real
  alternatives = cluster_to_background_emd(
    data, cluster_assignments, weights)
  n_clusters, n_genes = size(alternatives)
  sample_size = length(cluster_assignments)
  # `counts` needs a contiguous set of integral labels with arbitrary starts and ends
  # however, `countmap` works better when I need to index by actual labels.
  # sums up the weights rather than purely counting
  cluster_weights = ( length(weights) > 0 ?
    countmap(cluster_assignments, weights) :
    countmap(cluster_assignments) )
  shuffled_clusters = copy(cluster_assignments)
  weighed_weights = copy(weights)
  nulls = zeros(T, n_clusters, n_genes, n_null_draws)
  for t in 1:n_null_draws # why no convenient threaded map? so many random packages to comb through..
    shuffle!(shuffled_clusters)
    if length(weights) > 0
      shuffled_cluster_weights = countmap(
        shuffled_clusters, weights)
      for w in 1:sample_size
        original_cluster = cluster_assignments[w]
        shuffled_cluster = shuffled_clusters[w]
        weight_ratio = ( cluster_weights[original_cluster]
          / shuffled_cluster_weights[shuffled_cluster] )
        weighed_weights[w] = weights[w] * weight_ratio
      end
    end
    draw = cluster_to_background_emd(
      data, shuffled_clusters, weighed_weights)
    nulls[:, :, t] = draw
  end
  distribution = zeros(T, n_null_draws)
  probabilities = zeros(T, n_clusters, n_genes)
  for i in 1:n_clusters, j in 1:n_genes
    alternative = alternatives[i, j]
    distribution[:] = nulls[i, j, :]
    if alternative > 0 # !isapprox(alternative, 0)
      probability = alternative |> ecdf(distribution)
      probabilities[i, j] = 1 - probability
    else # P-value should be unit EVEN IF all nulls are also zero,
      #    in which case the `ecdf` above would give unit.
      probabilities[i, j] = 1
    end
  end
  probabilities
end
