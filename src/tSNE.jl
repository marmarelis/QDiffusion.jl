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

using PyCall
using Conda

tsne_is_initialized = false

# call this whilst compiling? `__init__` would be called at import, every time (and there could only be one in my entire module)
function init_tsne()
  this_dir = @__DIR__
  python_dir = this_dir * "/deformed-tSNE"
  pushfirst!(PyVector(pyimport("sys")."path"), python_dir) # to locate the local module
  # make sure to install cython and numpy, as this command will not automatically do it (bypasses pyproject.toml)
  pyimport_conda("Cython", "cython")
  pyimport_conda("numpy", "numpy")
  pyimport_conda("scipy", "scipy")
  pyimport_conda("sklearn", "scikit-learn>=0.23.1")
  python = pyimport("sys").executable
  lib_path = Conda.LIBDIR
  current_dir = pwd()
  cd(this_dir * "/deformed-tSNE") # does my working directory always begin at the module scope?
  #pyimport("setuptools.sandbox").run_setup("setup.py", ["build_ext", "--inplace"])
  command = addenv(`$python setup.py build_ext --inplace`,
    Dict("LD_LIBRARY_PATH" => lib_path)) # emulate the Conda environment
  run(command)
  cd(current_dir)
  global tsne_is_initialized
  tsne_is_initialized = true
end

# wouldn't it make sense to define separate types for first-order optimizers
# with structs to store state, like accumulated variance for ADAM?
# it appears that standard t-SNE employs a rather rudimentary learning schedule

# notes.
# * adding small Levy noise to MNIST, a Gaussian on the p-norm with tiny p (like 0.1) seems to perform best
# * closer to normalcy, I want to see if p slightly less than 2 is not as good as q slightly greater than 1
# * is there a technique, agnostic to a choice of clustering method, that simply scores how well points
#   belonging to the same label congregate together? I'd rather not bring the biases and nuances of a
#   particular cluster-identification algorithm into the mix

function embed_tsne(data::AbstractMatrix{T}; scale::Bool, norm_power::Union{T, Int} = 2,
    n_manifold_dims::Int, deformation::T, perplexity::T, deform_sum::Bool,
    inner_bandwidth::T = T(1)) where T <: Real
  if !tsne_is_initialized
    init_tsne()
  end
  n_dims, n_points = size(data)
  procedure = pyimport("deformed_tSNE")
  scales = scale ? ones(T, n_dims) : scale_kernel(deformation, data) # scale normally when `~deform_sum` ?
  scales .*= inner_bandwidth # no need to rescale afterwards, as distances (and t-SNE bandwidths) are all relative
  # the burden does not seem to be on the GC/allocator below, but rather the frequent calling
  # between Python and Julia. I should pre-compute the entire matrix instead.
  squared_displacement = zeros(T, n_dims, 1)
  function get_norm(a::AbstractVector{T}, b::AbstractVector{T})::T
    squared_displacement[:, 1] = abs.((a .- b) ./ scales) .^ norm_power
    output = (!deform_sum ? dropdims(sum(squared_displacement, dims=1), dims=1) :
      -q_sum(deformation, -squared_displacement))
    output[1] ^ (1/norm_power) # sadly, the sklearn api forces us to sqrt and back
  end
  model = procedure.TSNE(
    n_components=n_manifold_dims, deformation=deformation,
    perplexity=perplexity, square_distances=true, metric="precomputed", # could have been `get_norm`
    verbose=1) # can't have `init="pca"` when `metric="precomputed"`
  #squared_displacements = [ ((data[k, i] - data[k, j]) / scales[k]) ^ 2
  #  for k in 1:n_dims, i in 1:n_points, j in 1:n_points ]
  #flat_squared_displacements = reshape(squared_displacements, n_dims, :)
  #flat_squared_distances = -q_sum(deformation, -flat_squared_displacements)
  #distances = reshape(sqrt.(flat_squared_distances), n_points, n_points)
  n_threads = Threads.nthreads()
  buffer = zeros(T, n_dims, n_points, n_threads) # thread-local buffer
  function get_norms(a::AbstractVector{T}, b::AbstractMatrix{T})::AbstractVector{T}
    thread = Threads.threadid()
    squared_row = @view buffer[:, :, thread]
    squared_row[:, :] = - abs.((a .- b) ./ scales) .^ norm_power
    output = -(deform_sum ? q_sum(deformation, squared_row) :
      dropdims(sum(squared_row, dims=1), dims=1))
    output .^ (1/norm_power)
  end
  # assuming this distance is a proper metric (to be proven shortly,)
  # tree construction should not require the comprehensive calculation
  # of all dyadic distances as done here
  # also, q-Gaussian kernels are NOT rotationally invariant, and as such
  # should not be expected to yield similar results before and after PCA
  distance_time = @elapsed begin
    #distances = hcat(( get_norms(@view(data[:, i]), data)
    #  for i in 1:n_points )...)
    distances = zeros(T, n_points, n_points)
    Threads.@threads for col in 1:n_points
      input = @view data[:, col]
      distances[:, col] = get_norms(input, data)
    end
    # lend a hand to the binary search for optimal bandwidths by rescaling
    distances ./= mean(distances)
  end
  @info "computed distances in $distance_time seconds"
  distances |> Iterators.flatten |> collect |> describe
  # PyCall can translate arbitrary Julia functions to Python callables!
  model.fit_transform(distances)
end

using Clustering

function score_tsne_clusters(data::AbstractMatrix{T}, labels::AbstractVector{Int};
    scale::Bool, n_manifold_dims::Int, pq_pairs::Vector{Tuple{Union{T, Int}, T}},
    deform_sum::Bool, perplexity::T, n_trials::Int, inner_bandwidth::T = T(1)
    )::Matrix{T} where T <: Real
  n_labels = maximum(labels) - minimum(labels) + 1 # MNIST labels are 0-indexed
  @assert labels |> unique |> length == n_labels
  n_settings = length(pq_pairs)
  scores = zeros(T, n_trials, n_settings)
  for (setting, (norm_power, deformation)) in enumerate(pq_pairs)
    for trial in 1:n_trials
      embedding = embed_tsne(data;
        scale, n_manifold_dims, deformation, perplexity,
        norm_power, deform_sum, inner_bandwidth)
      clusters =  kmeans(embedding', n_labels) |> assignments
      score = randindex(clusters, labels)[1]
      scores[trial, setting] = score
    end
  end
  scores
end
