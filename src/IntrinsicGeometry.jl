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

module IntrinsicGeometry

include("kernels.jl")
include("explicit-diffusion.jl")

include("implicit-diffusion.jl")

include("cuda-kernels.jl")

include("preprocessing.jl")
include("emd.jl")
include("PHATE.jl")
include("introspection.jl")
include("tSNE.jl")
include("discrepancies.jl")
include("NMF.jl")
include("neighbor-prediction.jl")

export DeformedKernel, GaussianKernel, PossiblyDeformedKernel, AdaptiveKernel

export estimate_inner_bandwidth

export reduce_to_principal_components, find_dispersed_features, preprocess_cell_counts

export estimate_explicit_embedding, extend_explicit_embedding, bootstrap_explicit_embedding

export estimate_implicit_embedding, read_scipy_sparse_matrix

export normalize_diffusion, decompose_diffusion, embed_diffusion

export extract_landmarks, perform_metric_mds, leave_it_to_phate, reduce_to_sparse_principal_components,
  estimate_landmark_cluster_attribute_variances, estimate_cluster_attribute_variances

export cluster_to_background_emd, test_cluster_to_background_emd

export smooth_local_dimensionality, compare_transport_paths, tune_and_score_svm_on_explicit_embedding,
  tune_and_score_gp_on_explicit_embedding, tune_and_score_nn_on_explicit_embedding, load_keel_dataset

export init_tsne, embed_tsne, score_tsne_clusters

export estimate_ksd_statistic, estimate_mmd_statistic, par_estimate_mmd, par_estimate_independence_mmd,
  par_cluster_mmd, par_spatially_cluster_mmd, find_communities, construct_knn, construct_weighted_knn, construct_shared_nn

export estimate_nmf, par_estimate_nmf, regress_linear, par_regress_linear

end # module
