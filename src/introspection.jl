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

# The scope of this file is to implement various critics and evaluation schemes
# for the quality of a manifold embedding, including its various auxiliary
# purposes like teasing out a branching structure.

# Estimating local intrinsic dimensionality in a manifold.
# The embedding is assumed to be Euclidean.
# See "On Local Intrinsic Dimension Estimation and Its Applications" (2010)

using NearestNeighbors, ProgressMeter

"""
  Implements the MLE approach. Since we are referring to an embedding, the shape of the
  coordinate matrix should be (n_points x n_dims). The tree is assumed to be constructed
  on these points.
"""
function estimate_local_dimensionality(tree::NNTree,
    radius::T, coords::Matrix{T})::Vector{T} where T <: Real
  n_points, n_dims = size(coords)
  # alternative is k-NN with `n_neighbors+1`
  neighborhoods = inrange(tree, coords', radius, false) # `sortres` here is phony: it does not sort by distance, but rather by index.
  squared_distances::Vector{Vector{T}} = [
    #let neighbors = @view neighborhood[2:end]
    dropdims(sum(
      (coords[neighborhood, :] .- coords[i, :]') .^ 2,
      dims=2), dims=2) |> sort!
    for (i, neighborhood) in enumerate(neighborhoods) ] # n_points-vector of n_neighbors[i]-vectors
  dimensionalities = zeros(T, 0)
  @showprogress for i in 1:n_points
    these_squared_distances = @view squared_distances[i][2:end]
    log_distances = 0.5log.(these_squared_distances)
    n_neighbors = length(log_distances)
    if n_neighbors <= 2
      push!(dimensionalities, 0)
      continue
    end
    log_distance_sum = sum(log_distances)
    log_ratio = n_neighbors*log_distances[end] - log_distance_sum # sum_{j=1}^k log(T_k / T_j)
    dimensionality = (n_neighbors-2) ./ log_ratio
    push!(dimensionalities, dimensionality)
  end
  dimensionalities
end

function estimate_local_dimensionality(tree::NNTree,
    n_neighbors::Int, coords::Matrix{T})::Vector{T} where T <: Real
  n_points, n_dims = size(coords)
  neighborhoods, neighbor_distances = knn(tree, coords', n_neighbors+1, true)
  distances = @view hcat(neighbor_distances...)[2:end, :] # (n_neighbors x n_points)
  neighbors = @view hcat(neighborhoods...)[2:end, :] # (n_neighbors x n_points)
  log_distances = log.(distances)
  log_distance_sums = dropdims(sum(log_distances, dims=1), dims=1)
  log_ratios = n_neighbors*log_distances[end, :] .- log_distance_sums # sum_{j=1}^k log(T_k / T_j)
  dimensionalities = (n_neighbors-2) ./ log_ratios
end

using Statistics

function smooth_local_dimensionality(
    param::Union{T,Int}, n_voters::Int, coords::Matrix{T};
    leaf_size::Int = 100)::Vector{T} where T <: Real
  n_points, n_dims = size(coords)
  tree = KDTree(coords', Euclidean(), leafsize=leaf_size)
  dimensionalities = estimate_local_dimensionality(
    tree, param, coords)
  neighborhoods, _ = knn(tree, coords', n_voters, false)
  neighbors = hcat(neighborhoods...) # (n_voters x n_points)
  votes = dimensionalities[neighbors] # index by a matrix of integers?
  decisions = median(votes, dims=1)[:]
end