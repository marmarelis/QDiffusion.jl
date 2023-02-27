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

# the "kernels" in cuda-kernels.jl carries a dual meaning:
#   that of affinity kernel and gpu kernel

using CUDA

# q-sum on a matrix in CUDA? with warp-level and, broadly, thread-atomic operations.
# chop batches into blocks, and launch a few at a time? kernel deals with block-level..
# not sure if I will end up packing EVERYTHING into a single kernel..
#   might have one or two more stages like a CuArray sum

# TURBO mode would be to take in two lists of coordinate vectors, and in ONE kernel
#   (which could comprise other nested function calls) compute a pairwise matrix of
#   q-Gaussian kernel affinities, bypassing large intermediate matrices
# construct, by this fashion, huge affinity matrices block by block

const n_cuda_threads = 512

# https://people.maths.ox.ac.uk/gilesm/cuda/2019/lecture_04.pdf
# https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/

const full_lane_mask = 0xff_ff_ff_ff # hardcoded for 32 lanes
# note that in Julia, 0xff denotes a UInt8, 0xff_ff denotes uInt16, etc

# MUST launch threads in multiples of 32 so that all are ACTIVE.
# not all may even participate in the ballot.
# even so, MUST ensure n_dims is a multiple of 32 so that warps
#   don't cross over. pad with zeros.
# do nvidia gpus have a concept of cache coherency?
#   seems like array access within a block has roughly uniform cost.
#   however, entire blocks run on single "streaming multiprocessors,"
#   which do have their own cache. locality could matter across blocks.
function cuda_q_sum(result, intermediate, factor::T, values) where T
  # passing `@cuda threads=(x,y) ...` makes threadIdx 2D
  # won't use this to address `values` because we don't
  #   want to impede on the number of threads that can be launched
  #   based on the shape of `values`
  thread_id = threadIdx().x
  n_threads = blockDim().x
  n_dims, n_points = size(values)
  lane_id = laneid() # 1:32; would @assert if it weren't costly
  warp_id = div(thread_id-1, 32) + 1
  n_total = n_dims * n_points
  # have it this way in case `intermediate` is bigger than necessary
  n_point_warps = div(n_dims, 32)
  stride = n_threads
  start = thread_id
  stop = n_total
  @inbounds for index in start:stride:stop
    # whole warp will always land in the same point,
    #   since `stride` is a multiple of 32
    # modulus/remainder should be optimized, hopefully
    # otherwise consider having a counter instead
    (point_index, dim_index) = divrem(index-1, n_dims) .+ 1
    warp_index = (dim_index-1) >> 5 + 1 # divide by 32
    partial = values[dim_index, point_index]
    #lane_mask = vote_ballot_sync(full_lane_mask, true)
    # if we were to attempt accessing lanes off the mask,
    # our own `partial` would return, which isn't very useful here.
    for bit_flip in 0:4 # butterfly shuffling. I could do this (tree reduction) on the CPU too..
      flip_mask = UInt32(1) << bit_flip
      other = shfl_xor_sync(full_lane_mask, partial, flip_mask)
      partial = partial + other + factor*partial*other
    end
    # now all warp lanes see the same `partial.`
    # is there a sane way to have each lane that was made redundant (half are, every iteration)
    # go find work in other warps? create other pseudo-warps? would have to read memory off-warp..
    if lane_id == 1
      # intermediate is (n_dims/32) x n_points
      intermediate[warp_index, point_index] = partial
    end
  end
  # do all over again, this time over warps rather than lanes
  current_width = n_point_warps
  while current_width > 1
    last_width = current_width
    current_width = (current_width >> 1) + (current_width & 1) # round up
    stop = current_width * n_points
    # all threads must run the same number of iterations, and hence of sync_threads calls
    @inbounds for base_index in 0:stride:stop
      index = base_index + start
      (point_index, warp_index) = divrem(index-1, current_width) .+ 1
      other_warp_index = warp_index + current_width
      sync_threads()
      (index > stop) && continue
      first = intermediate[warp_index, point_index]
      if other_warp_index <= last_width # treat out-of-bounds as adding zero
        # with this, hope we don't need the widths to be powers of two
        second = intermediate[other_warp_index, point_index]
        partial = first + second + factor*first*second
        # now intermediate[1:current_width, :] contain the partial sums
        intermediate[warp_index, point_index] = partial
      end
    end
  end
  # threads that finish earlier than others wait here for the rest to catch up? hope that's the case
  sync_threads()
  stop = n_points
  # basically an array copy...
  @inbounds for index in start:stride:stop
    result[index] = intermediate[1, index]
  end
  return nothing
end

using StatsFuns: logsumexp # NOT the NNlib/Flux version, which makes allocations
using StaticArrays
# assumes `factor:=1-q` is negative, along with EVERY value,
#  resulting in every term being negative as well.
#  So we store the opposite/negated in the log space.
# I replicate the code above (rather than accepting a Val{::Bool})
#  in order to keep functionalities separate.
# I log-transform in here rather than outside because I touch
#  each input once (effectively) anyhow.
# Got this one running in ONE shot!!
function cuda_log_q_sum(result, intermediate, factor::T, values) where T
  thread_id = threadIdx().x
  n_threads = blockDim().x
  n_dims, n_points = size(values)
  lane_id = laneid()
  warp_id = div(thread_id-1, 32) + 1
  n_total = n_dims * n_points
  n_point_warps = div(n_dims, 32)
  stride = n_threads
  start = thread_id
  stop = n_total
  log_factor = log(-factor)
  # can I invent a faster method tailored for this three-way addition?
  # for now I'm going with the naive approach
  #q_combine = (x, y) -> logaddexp(
  #  logaddexp(x, y),
  #  log_factor + x + y )
  # this is more sophisticated than reduce(logaddexp, ...)
  q_combine = (x, y) -> logsumexp(
    @SVector [x, y, log_factor+x+y] )
  @inbounds for index in start:stride:stop
    (point_index, dim_index) = divrem(index-1, n_dims) .+ 1
    warp_index = (dim_index-1) >> 5 + 1 # divide by 32
    value = values[dim_index, point_index]
    partial = log(-value) # log(+/- 0) should be -Inf
    for bit_flip in 0:4 # butterfly shuffling.
      flip_mask = UInt32(1) << bit_flip
      other = shfl_xor_sync(full_lane_mask, partial, flip_mask)
      partial = q_combine(partial, other)
    end
    if lane_id == 1
      intermediate[warp_index, point_index] = partial
    end
  end
  current_width = n_point_warps
  while current_width > 1
    last_width = current_width
    current_width = (current_width >> 1) + (current_width & 1) # round up
    stop = current_width * n_points
    @inbounds for base_index in 0:stride:stop
      index = base_index + start
      (point_index, warp_index) = divrem(index-1, current_width) .+ 1
      other_warp_index = warp_index + current_width
      sync_threads()
      (index > stop) && continue
      first = intermediate[warp_index, point_index]
      if other_warp_index <= last_width
        second = intermediate[other_warp_index, point_index]
        partial = q_combine(first, second)
        intermediate[warp_index, point_index] = partial
      end
    end
  end
  sync_threads()
  stop = n_points
  @inbounds for index in start:stride:stop
    result[index] = intermediate[1, index]
  end
  return nothing
end

#const cuda_q_tolerances = ImmutableDict(Float32 => 1f-4, Float64 => 1e-7)
is_deformed(factor::Float32) = abs(factor) > 1e-2 # just for q-exp simplification, not the q-sum
is_deformed(factor::Float64) = abs(factor) > 1e-4

# would a series of array operations fuse just as well? `par` for parallelize
function par_q_diffusion(q::T, power::T, bandwidths::CuArray{T},
    distances::CuArray{T})::CuArray{T} where T
  factor = 1 - q
  exp_q_power = 1/factor # LLVM should be able to optimize this on its own..
  # `result` and `distances` should be able to be the same array!!
  result = map(bandwidths, distances) do bandwidth, distance
    argument = - (-distance / bandwidth) ^ power
    # shouldn't incur a loss of performance because it's so easily predictable 
    if is_deformed(factor)
      diffusion = (1 + factor*argument) ^ exp_q_power
    else
      diffusion = exp(argument)
    end
  end
end

function par_log_q_diffusion(q::T, power::T, log_bandwidths::CuArray{T},
    log_distances::CuArray{T})::CuArray{T} where T
  factor = 1 - q
  exp_q_power = 1/factor
  result = map(log_bandwidths, log_distances) do bandwidth, distance
    # `distance` is actually `log(-distance)`
    argument = -exp((distance - bandwidth) * power)
    if is_deformed(factor)
      diffusion = (1 + factor*argument) ^ exp_q_power # log1pexp version? not worth..
    else
      diffusion = exp(argument)
    end
  end
end

function cuda_pairwise_chunk(row_bandwidths, col_bandwidths, displacements, bandwidths, points,
    norm_power::T, chunk_size::Int, row_start::Int, col_start::Int) where T <: Real
  n_dims, n_points = size(points)
  start = threadIdx().x
  stride = blockDim().x
  outer_start = mod(start-1, chunk_size) + 1
  inner_start = div(start-1, chunk_size) + 1
  inner_stride = div(stride-1, chunk_size) + 1
  if inner_start == inner_stride # if we are last in the inner loop
    # (the only group of threads of which the outer loop isn't fully covered)
    outer_stride = mod(stride-1, chunk_size) + 1
  else
    outer_stride = chunk_size # else we are done after one shot.
    # since we are idempotent, we could have omitted this conditional
    # and simply allowed redundant operations on latter columns.
  end
  # memory access would be more efficient if we strided along this outer loop,
  # but `chunk_size` may be smaller than the number of CUDA threads (`stride`).
  # if chunk size were guaranteed to be a divisor of stride,
  # then we could enact some trickery by striding both `col` and `row`.
  # I'm being even trickier now: with this implementation,
  # we simply require that n_threads < chunk_size^2.
  @inbounds for col in outer_start:outer_stride:chunk_size
    col_offset = (col-1) * chunk_size
    col_index = col + col_start-1
    for row in inner_start:inner_stride:chunk_size
      index = row + col_offset
      row_index = row + row_start-1
      row_bandwidths[index] = bandwidths[row_index]
      col_bandwidths[index] = bandwidths[col_index]
      for dim in 1:n_dims # start:stride:n_dims
        if (row_index > n_points) || (col_index > n_points)
          # unimportant because we ignore all computations involving out-of-bounds
          displacements[dim, index] = T(0)
        else
          difference = points[dim, row_index] - points[dim, col_index]
          # since we q-sum the negative part
          displacement = - abs(difference) ^ norm_power
          displacements[dim, index] = displacement
        end
      end
    end
  end
  return nothing
end

using ProgressMeter
using Base.Threads
using LinearAlgebra

# construct a sparse matrix chunk by chunk?
function par_pairwise_q_diffusion(q::T, bandwidths::AbstractVector{T},
    points::AbstractMatrix{T}, chunk_size::Int, in_log::Bool;
    dim_scales=ones(T, 1), local_scales=ones(T, 1), do_deform_norm::Bool=true, # `dim_scales` should absorb the inner bandwidth
    norm_power::T=T(2), decay_power::T=norm_power, do_distances::Bool=false,
    devices::Vector{Int}, verbose::Bool=false)::Matrix{T} where T
  n_threads = nthreads()
  if length(devices) == 1
    devices = fill(devices[1], n_threads)
  elseif length(devices) == 0
    devices = 0:(n_threads-1) |> collect
  end
  # each thread needs to be given a device to use.
  # for a particularly large GPU, you can assign it multiple times.
  # each thread launches a task, with a dedicated context for `device!(.)`
  @assert length(devices) == n_threads
  points = (points ./ dim_scales) ./ local_scales'
  n_dims, n_points = size(points)
  @assert length(bandwidths) == n_points
  inter_size = chunk_size ^ 2
  factor = 1 - q
  padded_n_dims = n_dims + 31 - ((n_dims-1) % 32)
  # make this a Val{Log} and have it fold statically?
  (dyn_q_sum, dyn_q_diffusion) = ( in_log ?
    (cuda_log_q_sum, par_log_q_diffusion) :
    (cuda_q_sum, par_q_diffusion) )
  @assert padded_n_dims >= n_dims && padded_n_dims % 32 == 0
  per_device = f -> map(devices|>enumerate) do (i, d)
    device!(d); f(i)
  end
  points_d = per_device() do _
    points |> CuArray{T}
  end
  bandwidths_d = per_device() do _
    bandwidths |> CuArray{T}
  end
  # we will always fill in just 1:n_dims of the 1:padded_n_dims entries
  # so the rest shall remain zeros
  displacements_d = per_device() do _
    CUDA.zeros(T, padded_n_dims, inter_size)
  end
  row_bandwidths_d = per_device() do _
    CuArray{T}(undef, inter_size) # bandwidths to go along with the chunk,
  end
  col_bandwidths_d = per_device() do _
    CuArray{T}(undef, inter_size) # row-wise and col-wise
  end
  intermediate_d = per_device() do _
    CuArray{T}(undef, div(padded_n_dims, 32), inter_size) # +1 for good measure?
  end
  result_d = per_device() do _
    CuArray{T}(undef, inter_size)
  end
  chunk = per_device() do _ # not actually GPU-residing but convenient nonetheless
    zeros(T, inter_size)
  end
  mem_lock = SpinLock()
  affinities = zeros(T, n_points, n_points)
  n_iterations = div((div(n_points, chunk_size) + 1) ^ 2, 2)
  progress = Progress(n_iterations,
    desc="Processing chunks... ", enabled=verbose)
  # todo more memory-efficient parcelization into chunks that are contiguous when flattened
  @threads :static for col in 1:chunk_size:n_points # multithread here as a way of delegating concurrent GPU tasks?
    thread = threadid()
    # I guess Julia uses the Task scheduler for threads because any yielding/blocking
    # operation can land the existing task in a separate thread. this is clearly a problem
    # here if we keep the old `thread` handle and another thread starts using it too.
    # Julia's multithreading examples are really misleading about this, even though the technical docs warn about it.
    # the `:static` scheduler option (rather than the `:dynamic` default) fixes this. I wonder if this change in
    # behavior was introduced somewhere between Julia 1.6 and Julia 1.8. I hadn't seen this problem before.
    devices[thread] |> device!
    row_bandwidths_t = row_bandwidths_d[thread]
    col_bandwidths_t = col_bandwidths_d[thread]
    displacements_t = displacements_d[thread]
    bandwidths_t = bandwidths_d[thread]
    points_t = points_d[thread]
    result_t = result_d[thread]
    intermediate_t = intermediate_d[thread]
    # `Symmetric` matrices use the upper triangular part --- fill that in
    first_row = col
    for row in first_row:chunk_size:n_points
      @cuda threads=n_cuda_threads cuda_pairwise_chunk(
        row_bandwidths_t, col_bandwidths_t, displacements_t,
        bandwidths_t, points_t, norm_power, chunk_size, row, col)
      if do_deform_norm
        @cuda threads=n_cuda_threads dyn_q_sum(
          result_t, intermediate_t, factor, displacements_t)
      else
        # if this doesn't fully fuse and vectorize, I should just create a new CuArray each time
        result_t[:] .= .-sum(displacements_t, dims=1)[:]
      end
      if !do_distances # the "normal" path
        power = decay_power / norm_power
        #lock(mem_lock) do # this is to preempt interference with other threads' CUDA GCs
          row_diffusions_t = dyn_q_diffusion(q, power, row_bandwidths_t, result_t)
          col_diffusions_t = dyn_q_diffusion(q, power, col_bandwidths_t, result_t)
          diffusions_t = (row_diffusions_t .+ col_diffusions_t) ./ 2
          copyto!(chunk[thread], diffusions_t)
          # free AFTER read.
          (row_diffusions_t, col_diffusions_t, diffusions_t) .|> CUDA.unsafe_free!
          # hope this gets that sum from the else-statement above
          # manual memory management by necessity (only in the multi-device setting)..
          CUDA.reclaim() # does it slow things down?
        #end # does this lock make things too slow? the cost of safety.. maybe I can bear the occasional crash
      else # bypass the above for "simple" distance calculations
        copyto!(chunk[thread], result_t)
        # if in_log, keep the bandwidths as log-minus distances?
      end
      # reshape might return something that looks like it was copied, but it does not
      square_chunk = reshape(chunk[thread], chunk_size, chunk_size)
      row_range = row : min(row+chunk_size-1, n_points)
      col_range = col : min(col+chunk_size-1, n_points)
      # allow progressive construction of sparse matrix by appending nonzeros to lists and then sorting?
      affinities[row_range, col_range] =
        @view square_chunk[1:length(row_range), 1:length(col_range)]
      next!(progress) # inner loop for granular progress
    end
  end
  per_device() do i
    ( points_d[i], bandwidths_d[i], displacements_d[i],
      row_bandwidths_d[i], col_bandwidths_d[i], intermediate_d[i],
      result_d[i] ) .|> CUDA.unsafe_free!
    CUDA.reclaim() # be respectful and clean up after yourself.
    # otherwise CUDA might run into issues while (quite recklessly)
    # freeing variables on different devices
  end # we're filling in the lower triangular part actually (better for memory? maaybe)
  affinities' |> Symmetric |> Matrix
end


const default_chunk_size = 64 # or 48^2 = 2304 is a good amount

function par_evaluate_pairwise(kernel::DeformedKernel{T}, bandwidth::T,
    points::AbstractMatrix{T}, devices::Vector{Int};
    chunk_size::Int=default_chunk_size, in_log::Bool=true,
    point_indices::AbstractVector{Int}=axes(points, 2),
    verbose::Bool=true)::Matrix{T} where T
  n_dims, n_points = size(points)
  q = kernel.deformation
  dim_scales = kernel.scales * kernel.inner_bandwidth
  norm_power = kernel.norm_power
  decay_power = kernel.decay_power
  do_deform_norm = kernel.do_deform_norm
  bandwidths = if in_log
    fill(log(bandwidth / kernel.inner_bandwidth) * norm_power, n_points)
  else
    ( fill(bandwidth, n_points)
    ./ kernel.inner_bandwidth ^ norm_power )
  end
  par_pairwise_q_diffusion(q, bandwidths, points, chunk_size, in_log; devices,
    dim_scales, norm_power, decay_power, do_deform_norm, verbose)
end

# we specialize only some particular kernels: namely the most useful ones.
# any use for `local_scales`? when I find one is when I'll expose the functionality
function par_evaluate_pairwise(wrapper::AdaptiveKernel{T,DeformedKernel{T}},
    bandwidth::T, points::AbstractMatrix{T}, devices::Vector{Int};
    # indices of the points here wrt the batch originally given to the kernel
    point_indices::AbstractVector{Int}=axes(points, 2),
    chunk_size::Int=default_chunk_size, verbose::Bool=true
    )::Matrix{T} where T
  in_log = wrapper.in_log
  n_dims, n_points = size(points)
  kernel = wrapper.inner_kernel
  q = kernel.deformation
  dim_scales = kernel.scales * kernel.inner_bandwidth
  norm_power = kernel.norm_power
  decay_power = kernel.decay_power
  do_deform_norm = kernel.do_deform_norm
  bandwidths = if in_log
    ( wrapper.nearest_neighbors[point_indices]
    .+ log(bandwidth / kernel.inner_bandwidth) * norm_power )
  else
    ( wrapper.nearest_neighbors[point_indices]
    .* bandwidth ./ kernel.inner_bandwidth ^ norm_power )
  end
  par_pairwise_q_diffusion(q, bandwidths, points, chunk_size, in_log; devices,
    dim_scales, norm_power, decay_power, do_deform_norm, verbose)
end

# specialize the AdaptiveKernel constructor too
# `in_log:=true` is overall decently slower but numerically stable
function AdaptiveKernel(neighbor_rank::Int, inner_kernel::DeformedKernel{T},
    points::AbstractMatrix{T}; cuda_devices::Vector{Int},
    chunk_size::Int=default_chunk_size, in_log::Bool, verbose::Bool=true) where T
  n_dims, n_points = size(points)
  inner_bandwidth = inner_kernel.inner_bandwidth
  q = inner_kernel.deformation
  dim_scales = inner_kernel.scales * inner_bandwidth
  norm_power = inner_kernel.norm_power
  decay_power = inner_kernel.decay_power
  do_deform_norm = inner_kernel.do_deform_norm
  bandwidths = ones(T, n_points) # shouldn't be used in this pathway
  distances = par_pairwise_q_diffusion(q, bandwidths, points,
    chunk_size, in_log; dim_scales, norm_power, decay_power, do_deform_norm,
    devices=cuda_devices, verbose, do_distances=true )
  if in_log
    distances .+= log(inner_bandwidth) * norm_power
  else
    distances .*= -inner_bandwidth ^ norm_power
  end
  sorted_distances = sort!(distances, dims=1)
  nearest_neighbors = sorted_distances[neighbor_rank+1, :]
  AdaptiveKernel(inner_kernel, neighbor_rank, nearest_neighbors, in_log)
end
