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

# one-sample and two-sample statistics: KSD and MMD, respectively
# Kernelized Stein Discrepancy---Liu et al. (2016)
# Maximum Mean Discrepancy---described in the same paper and many others, of course
# test on distributions (could be mixtures) that are LOCALLY non-Gaussian

@enum StatisticType U V

# NOTE FOR ELSEWHERE -- `rand` accesses a global state and risks serious data races when multithreaded

function compute_statistic(statistic::Function, type::StatisticType,
    first_sample::AbstractMatrix{T}, second_sample::AbstractMatrix{T}=first_sample
    )::T where T <: Real
  n_dims, sample_size = size(first_sample)
  @assert all(size(first_sample) .== size(second_sample))
  sums = zeros(T, sample_size)
  Threads.@threads for i in 1:sample_size
    for j in 1:sample_size
      if type == U && i == j
        continue
      end
      first = @view first_sample[:, i]
      second = @view second_sample[:, j]
      sums[i] += statistic(first, second)
    end
  end
  total = sum(sums)
  if type == U
    total / (sample_size * (sample_size-1))
  else
    total / sample_size^2
  end
end

function simple_kernel_evaluation(kernel::Kernel{T}, bandwidth::T,
    first::AbstractVector{T}, second::AbstractVector{T})::T where T <: Real
  displacements = hcat(first .- second) # matrix with singleton column
  # no support for kernels relying on point indices, for now
  evaluate(kernel, bandwidth, displacements, one(T), [(0, 0)])[1]
end

using Zygote, Flux, FiniteDifferences

# u_q(x,x') as defined by Liu et al. (2016). see also Oates, Girolami, Chopin (BIG SHOTS)
function evaluate_stein_distance(log_likelihood::Function, kernel::Kernel{T}, bandwidth::T,
    first::AbstractVector{T}, second::AbstractVector{T}; n_grid_points::Int=3)::T where T <: Real
  n_dims = length(first)
  @assert length(second) == n_dims
  first_gradient, = gradient(log_likelihood, first)
  second_gradient, = gradient(log_likelihood, second)
  first_kernel_gradient, = gradient(first) do point
    simple_kernel_evaluation(kernel, bandwidth, point, second)
  end
  second_kernel_gradient, = gradient(second) do point
    simple_kernel_evaluation(kernel, bandwidth, first, point)
  end
  kernel_value = simple_kernel_evaluation(kernel, bandwidth, first, second)
  first_term = first_gradient' * kernel_value * second_gradient
  second_term = first_gradient' * second_kernel_gradient
  third_term = first_kernel_gradient' * second_gradient
  # a rather roundabout way for finding the trace of the Hessian, basically
  # is this more efficient than computing the Hessian, straight up?
  # Zygote complains about the indexing with "mutating arrays is not supported"
  # from what I understand, Zygote is a ReverseDiff with IR-level rewriting
  fourth_term = sum(1:n_dims) do i
    new_first = copy(first)
    central_fdm(n_grid_points, 1)(first[i]) do slider
      new_first[i] = slider
      g, = gradient(second) do second
        #second = @set second[i] = entry # overwrite without mutating
        simple_kernel_evaluation(kernel, bandwidth, new_first, second)
      end
      g[i]
    end
  end
  first_term + second_term + third_term + fourth_term
end

function estimate_ksd_statistic(log_likelihood::Function, kernel::Kernel{T},
    bandwidth::T, sample::AbstractMatrix{T})::T where T <: Real
  compute_statistic(U, sample) do first, second
    evaluate_stein_distance(log_likelihood, kernel, bandwidth, first, second)
  end
end

# not the most efficient implementation... no batching whatsoever
function estimate_mmd_statistic(kernel::Kernel{T}, bandwidth::T,
    first_sample::AbstractMatrix{T}, second_sample::AbstractMatrix{T})::T where T <: Real
  eval_kernel(first, second) = simple_kernel_evaluation(kernel, bandwidth, first, second)
  first_cohesion = compute_statistic(eval_kernel, U, first_sample)
  second_cohesion = compute_statistic(eval_kernel, U, second_sample)
  interplay = compute_statistic(eval_kernel, V, first_sample, second_sample)
  first_cohesion + second_cohesion - 2interplay
end

# newer CUDA version
# use the first hypothesis test on the biased MMD, due to Gretton et al. (2012) 
function par_estimate_mmd(make_kernel::Function, bandwidth::T,
    first_sample::AbstractMatrix{T}, second_sample::AbstractMatrix{T};
    make_psd::Bool=false, cuda_devices::Vector{Int}, confidence::T=T(0.05),
    verbose::Bool=true)::NamedTuple where T <: Real
  @assert CUDA.functional()
  points = hcat(first_sample, second_sample)
  n_dims, n_first = size(first_sample)
  _, n_second = size(second_sample)
  kernel = make_kernel(points) :: Kernel{T}
  affinities = par_evaluate_pairwise(
    kernel, bandwidth, points, cuda_devices; verbose)
  if make_psd
    bump = minimum(eigvals(affinities) .|> real) |> T
    affinities -= I(n_first+n_second) * bump
  end
  first = mean( @view affinities[1:n_first, 1:n_first] )
  second = mean( @view affinities[n_first+1:end, n_first+1:end] )
  cross = mean( @view affinities[1:n_first, n_first+1:end] )
  statistic = max(0, first + second - 2cross) |> sqrt
  magnitude = maximum(affinities|>diag)
  # the test assumes these are equal. we do a worst-case
  degrees = min(n_first, n_second) 
  # acceptance (critical?) region. harder to solve for the P-value
  acceptance = sqrt(2magnitude/degrees) * ( 1 + sqrt(-2log(confidence)) )
  null = statistic < acceptance
  (; statistic, null )
end

# dependence between sets of genes from the same cells? take a pathway database?
function par_estimate_independence_mmd(make_kernel::Function, bandwidth::T,
    first_sample::AbstractMatrix{T}, second_sample::AbstractMatrix{T};
    make_psd::Bool=false, cuda_devices::Vector{Int}, confidence::T=T(0.05),
    verbose::Bool=true)::NamedTuple where T <: Real
  paired_points = vcat(first_sample, second_sample)
  n_first_dims, n_points = size(first_sample)
  n_second_dims, _ = size(second_sample)
  scrambled_second_sample = @view second_sample[:, shuffle(1:n_points)]
  # could also bootstrap both first and second for the unpaired points?
  # permutation vs bootstrap.. sample effects? motivation is less clear than with directly simulating a null
  unpaired_points = vcat(first_sample, scrambled_second_sample)
  par_estimate_mmd(make_kernel, bandwidth, paired_points, unpaired_points;
    make_psd, cuda_devices, confidence, verbose)
end

abstract type GridMethod end

struct RectangularGrid <: GridMethod
  min_size :: Int
end

using Distributions, CategoricalArrays # for `cut` discretization/binning
function discretize_grid(method::RectangularGrid,
    coordinates::AbstractMatrix{T}, n_bins::Int
    )::Vector{Int} where T
  (; min_size ) = method
  n_dims, n_points = size(coordinates)
  # draw a grid along the principal components.
  components = reduce_to_principal_components(
    coordinates, n_dims )
  aligned = components.reduced
  scatters = sqrt.(components.eigenvalues) # sqrt of variances along axes
  n_partitions = floor.(Int, # per dimension, preserving aspect ratios and factorizing n_bins
    scatters  .*  (n_bins / prod(scatters)) ^ (1/n_dims) )
  grid = mapreduce(hcat, axes(aligned, 1)) do dim
    values = @view aligned[dim, :]
    left, right = minimum(values), maximum(values)
    partitions = range(left-eps(left), stop=right+eps(right),
      length=n_partitions[dim]+1 )
    cut(values, partitions) .|> levelcode
  end # points x dims
  if min_size > 0
    done = false
    # once a label becomes "okay" it shall remain okay
    okay = Dict{SVector{n_dims,Int}, Bool}() # converts every key that comes in
    while !done
      done = true
      for point_index in axes(grid, 1)
        point = @view grid[point_index, :]
        get(okay, point, false) && continue # cache to speed up
        n_others = mapslices(grid, dims=2) do other
          all( point .== other )
        end |> sum
        if n_others >= min_size
          okay[point] = true
          continue
        end
        done = false
        # merge along the smallest axis, probabilistically
        i = min(rand(Geometric(0.9)) + 1, n_dims) # fuzziness prevents deadlocks
        if point[i] == n_partitions[i]
          point[i] -= 1
        elseif point[i] == 1
          point[i] += 1
        else
          point[i] += sample([-1, +1])
        end
      end
    end
  end
  powers = vcat(1, n_partitions[begin:end-1]) |> cumprod
  pixels = mapslices(grid, dims=2) do levels
    sum( powers .* (levels.-1) ) + 1
  end
  # flooring `n_partitions` ensures that the below will never be violated
  @assert all( (pixels .> 0) .& (pixels .<= n_bins) )
  pixels = indexin(pixels[:], sort(pixels|>unique)) # flatten and collapse missing entries
end

struct HierarchicalGrid <: GridMethod end

function discretize_grid(::HierarchicalGrid,
    coordinates::AbstractMatrix{T}, n_bins::Int
    )::Vector{Int} where T
  cutree(
    hclust(
      pairwise(Euclidean(), coordinates),
      linkage=:ward ),
    k=n_bins )
end

# turns out k-means enjoys a much tighter theoretical relation
# to Voronoi diagrams, and finding partitions of polyhedra(?)
# that are roughly of equal (spatial) volume, rather than equal
# number of points that the hierarchical above roughly accomplishes.
struct VoronoiGrid <: GridMethod
  n_trials :: Int
end
# also look up "fair polygon partitioning" problem
function discretize_grid(method::VoronoiGrid,
    coordinates::AbstractMatrix{T}, n_bins::Int
    )::Vector{Int} where T
  (; n_trials ) = method
  attempts = map(1:n_trials) do trial
    kmeans(coordinates, n_bins, init=:kmpp) |> assignments
  end
  fitnesses = map(attempts) do attempt
    counts(attempt) |> minimum
  end
  attempts[ argmax(fitnesses) ]
end


# the adaptive nature of some of these kernels means that you will get different 
# affinities for the same pair depending on the surrounding points in the sample.
# currently, each pair only considers the union of that pair.
function par_spatially_cluster_mmd(make_kernel::Function, bandwidth::T,
    coordinates::AbstractMatrix{T}, genes::AbstractMatrix{T};
    n_pixels::Int, method::GridMethod, make_psd::Bool=false,
    cuda_devices::Vector{Int} )::NamedTuple where T <: Real
  # contrary to what I thought previously,
  # the centrality-initialized kmeans (kmcen) is not fully deterministic
  #  pixels = kmeans(coordinates, n_pixels, init=:kmpp) |> assignments
  # if we want more even clustering than kmeans gives
  pixels = discretize_grid(method, coordinates, n_pixels)
  distances = par_cluster_mmd(make_kernel, bandwidth, genes, pixels;
    make_psd, cuda_devices)
  (; pixels, distances )
end

function par_cluster_mmd(make_kernel::Function, bandwidth::T,
    genes::AbstractMatrix{T}, labels::Vector{Int};
    make_psd::Bool=false, cuda_devices::Vector{Int}
    )::AbstractMatrix{T} where T <: Real
  @assert minimum(labels) == 1
  n_labels = maximum(labels)
  distances = zeros(Float32, n_labels, n_labels)
  # sometimes it's just not worth the kludginess of map(Iterators.product(...))
  progress = Progress(div(n_labels * (n_labels-1), 2),
    desc="Comparing pairs of labels... ", enabled=true)
  for first in 1:n_labels, second in 1:(first-1) # diagonal is zero by definition
    first_genes = @view genes[:, labels.==first]
    second_genes = @view genes[:, labels.==second]
    distance = par_estimate_mmd(
      make_kernel, bandwidth, first_genes, second_genes;
      make_psd, cuda_devices, verbose=false ).statistic
    distances[second, first] = distance
    @assert isfinite(distance)
    next!(progress)
  end
  Symmetric(distances)
end

# Seurat does it this way (with the older Louvain algorithm, it seems)
function construct_knn(
    distances::AbstractMatrix{T}, n_neighbors::Int
    )::Matrix{Bool} where T
  n_nodes = size(distances, 1)
  @assert n_nodes == size(distances, 2)
  adjacency = zeros(Bool, n_nodes, n_nodes)
  @threads for node in 1:n_nodes
    # skip first because we don't want self-loops
    neighbors = @views sortperm(distances[:, node])[2:n_neighbors]
    for other in neighbors
      adjacency[other, node] = true
    end
  end
  adjacency |> permutedims # transposed was more cache-friendly
end

# in addition to this being Scanpy's cutting-edge approach
# for neighborhood graph construction, it seems that Monocle3
# also does a basic version of this (but they actually DO UMAP?
# and then use the embedding coordinates? I thought that was
# forsaken to purely visualization purposes)
function construct_umap_knn(
    points::AbstractMatrix{T}, n_neighbors::Int,
    metric::String="euclidean", seed::Int=42)::Matrix{Bool} where T
  n_dims, n_points = size(points)
  scanpy = pyimport("scanpy")
  indices, = scanpy.neighbors.compute_neighbors_umap(
    points', n_neighbors; metric, random_state=seed )
  adjacency = zeros(Bool, n_points, n_points)
  @threads for node in 1:n_points
    neighbors = @view indices[node, 2:n_neighbors]
    for other in neighbors
      adjacency[other+1, node] = true
    end
  end
  adjacency |> permutedims
end

function construct_weighted_knn(
    distances::AbstractMatrix{T}, n_neighbors::Int
    )::Matrix{T} where T
  n_nodes = size(distances, 1)
  @assert n_nodes == size(distances, 2)
  adjacency = zeros(T, n_nodes, n_nodes)
  @threads for node in 1:n_nodes
    neighbors = @views sortperm(distances[:, node])[2:n_neighbors] # no begin+1 here
    for (rank, other) in enumerate(neighbors)
      adjacency[other, node] = 1 / (1-rank) #decay ^ (1-rank)
    end
  end
  adjacency |> permutedims # transposed was more cache-friendly
end

# can be weighted or boolean. we implement a sort of fuzzy Jaccard. edges should be in [0,1]
# as described by Seurat, the existing kNN graph's edges are refined
function construct_shared_nn(adjacency::AbstractMatrix{T};
    cutoff::Float64, threshold::Float64=1e-3)::AbstractMatrix{Float32} where T
  n_points = size(adjacency, 1)
  @assert n_points == size(adjacency, 2)
  snn = zeros(Float32, n_points, n_points)
  Threads.@threads for i in axes(adjacency, 1)
    first = adjacency[i, :] # implicit copies
    first[i] = 1 # always include self, regardless of whether adjacency includes it
    for j in findall(first .> threshold)
      second = adjacency[j, :]
      second[j] = 1
      edge = sum( min.(first, second) ) / sum( max.(first, second) )
      snn[j, i] = edge
    end
  end
  ifelse.(snn .< cutoff, 0f0, snn) |> permutedims
end

function find_communities(adjacency::AbstractMatrix{T};
    seed::Int=42, resolution::Float64=NaN )::Vector{Int} where T
  ig = pyimport_conda("igraph", "igraph")
  la = pyimport("leidenalg")
  # `adjacency` can be Bools or Floats for weighted directed graphs (like if you used a kernel directly)
  # weirdly, igraph's Adjacency constructor doesn't recognize non-floats
  # row to column. igraph documentation is TERRIBLE. Julia's ecosystem puts it to shame.
  constructor = T <: Integer ? ig.Graph.Adjacency : ig.Graph.Weighted_Adjacency
  graph = constructor(Float64.(adjacency), mode="directed")
  # could also use the "CPM" <-> "Modularity" quality function.
  # these others take a `resolution_parameter`
  partition = if isfinite(resolution)
    la.find_partition(graph, la.CPMVertexPartition; seed,
      resolution_parameter=resolution)
  else
    la.find_partition(graph, la.ModularityVertexPartition; seed)
  end
  partition.membership .+ 1 # one-based indexing
end

# feed in normalized cells for roughly the method in 
# "Optimal transport improves cell–cell similarity inference in single-cell omics data"
function find_transport_distance_matrix(data::AbstractMatrix{T},
    prior_distances::AbstractMatrix{T}, reg::T; norm::T=T(2),
    n_neighbors::Int, verbose::Bool=true)::AbstractMatrix{T} where T
  n_dims, n_points = size(data)
  transposed = permutedims(data) # way faster this way
  gene_distances = zeros(T, n_dims, n_dims)
  progress = Progress(div((n_dims-1) * (n_dims-2), 2),
    desc="Gene-gene distances...", enabled=verbose)
  Threads.@threads for j in 2:n_dims
    for i in 1:(j-1)
      gene_distances[i, j] = mean(1:n_points) do k # `mean` just for numerical niceties
        abs(transposed[k, i] - transposed[k, j]) ^ norm
      end ^ (1/norm)
      next!(progress)
    end
  end
  gene_distances = Symmetric(gene_distances) |> Matrix
  normalized_data = data ./ sum(data, dims=1)
  cell_distances = fill(T(Inf), n_points, n_points)
  progress = Progress(n_points,
    desc="Cell-cell $n_neighbors-NN distances...", enabled=verbose)
  for j in 1:n_points
    cell_distances[j, j] = T(0)
    indices = @views sortperm(prior_distances[:, j])
    neighbors = @view indices[2:n_neighbors]
    distances = sinkhorn_divergence( # nice: it can broadcast!
      normalized_data[:, neighbors], normalized_data[:, j],
      gene_distances, reg )
    for (index, i) in enumerate(neighbors)
      # always fill in the upper triangular part
      cell_distances[min(i, j), max(i, j)] = distances[index]
    end
    next!(progress)
  end
  Symmetric(cell_distances)
end