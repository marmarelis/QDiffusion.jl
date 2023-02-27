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

using Statistics
using LinearAlgebra


# similar to the Karhunen-Loeve transform
function reduce_to_principal_components(data::AbstractMatrix{<:Real}, n_dims::Int)
  eigval, eigvec = data' |> cov |> Hermitian |> eigen
  means = mean(data, dims=2)
  n_dims = min(n_dims, size(data, 1))
  projection = eigvec[:, (end-n_dims+1):end]'
  reduced = projection * (data .- means)
  eigenvalues = eigval[(end-n_dims+1):end]
  (; reduced, projection, means, eigenvalues)
end

using Loess
# Seurat3 style, "vst" method, all defaults
# raw count data expected
function find_dispersed_features(data::AbstractMatrix, span::Float64=0.3)
  n_dims, n_points = size(data)
  clip = sqrt(n_points)
  means = mean(data, dims=2)[:]
  variances = var(data, dims=2)[:]
  inputs = log10.(means) .|> Float64
  outputs = log10.(variances) .|> Float64
  fit = loess(inputs, outputs; span)
  expected_variances = 10 .^ Loess.predict(fit, inputs)
  standardized = (data .- means) ./ sqrt.(expected_variances)
  min.( std(standardized, dims=2)[:], clip )
end

# using all the defaults for our project
function preprocess_cell_counts(data::AbstractMatrix)
  dispersion = find_dispersed_features(data)
  normalized = log1p.( 1e4 * data ./ sum(data, dims=1) ) |> Matrix{Float32}
  seurat_selected_indices = sortperm(-dispersion)[1:2_000]
  seurat_selected = normalized[seurat_selected_indices, :]
  seurat_standardized = mapslices(zscore, seurat_selected, dims=2)
  seurat = reduce_to_principal_components(seurat_standardized, 16)
  monocle = reduce_to_principal_components(normalized, 16)
  (; normalized, seurat, monocle )
end

using Clustering, StatsBase

# with defaults; https://github.com/hemberg-lab/SC3/blob/master/R/CoreMethods.R
function compute_sc3_clusters(data::AbstractMatrix{T}, ks=2:5, seed=1337) where T
  Random.seed!(seed)  
  dropout = mean(data .< 1e-3, dims=2)[:]
  filtered = (dropout .>= 0.1) .& (dropout .<= 0.9)
  filtered_data = data[filtered, :]
  n_dims, n_points = size(filtered_data)
  eigen_range = round.(Int, range(0.04n_dims, stop=0.07n_dims, length=15)) |> unique
  @info "filtered down to $(sum(filtered)) genes"
  euclidean = pairwise(Euclidean(), filtered_data)
  @info "computed Euclidean"
  pearson = 1 .- cor(filtered_data)
  @info "computed Pearson"
  spearman = 1 .- corspearman(filtered_data)
  @info "computed Spearman"
  components = map([euclidean, pearson, spearman]) do dist
    reduce_to_principal_components(dist, eigen_range[end]).projection
  end
  laplacians = map([euclidean, pearson, spearman]) do dist
    affinities = exp.(.-dist ./ maximum(dist))
    scales = sum(affinities, dims=2) .^ T(-0.5)
    affinities .*= scales .* scales'
    laplacian = I(n_points) - affinities
    eigval, eigvec = laplacian |> Hermitian |> eigen
    eigvec'
  end
  @info "computed transformations"
  clusters = map(Iterators.product( vcat(components, laplacians), eigen_range, ks )
      ) do (coords, n_eigen, n_clusters)
    reduced = coords[(end-n_eigen+1):end, :]
    kmeans(reduced, n_clusters) |> assignments
  end
  consensus = zeros(n_points, n_points, length(ks))
  for k_index in 1:length(ks)
    for grouping in clusters[:, :, k_index]
      for j in 1:n_points, i in 1:n_points
        if grouping[i] == grouping[j]
          consensus[i, j, k_index] += 1
        end
      end
    end
  end
  @info "computed consensi"
  consensus ./= 6length(eigen_range)
  clusters = Vector{Int}[]
  scores = Float64[]
  for (k_index, k) in enumerate(ks)
    distances = pairwise(Euclidean(), @view consensus[:, :, k_index]) # hmm, curious...
    hierarchy = hclust(distances, linkage=:complete)
    clustering = cutree(hierarchy; k)
    score = silhouettes(clustering, distances) |> mean
    push!(clusters, clustering)
    push!(scores, score)
  end
  clusters[argmax(scores)]
end