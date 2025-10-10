// diff_gaussian_rasterization/exist_prob_pruner.h

#ifndef EXIST_PROB_PRUNER_H
#define EXIST_PROB_PRUNER_H

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> compute_distances(
    torch::Tensor points,
    int k,
    float max_radius);


torch::Tensor compute_local_probs(
    torch::Tensor existence_probs,
    torch::Tensor alpha,
    torch::Tensor age_weight,
    torch::Tensor view_weight,
    torch::Tensor sh_weight,
    torch::Tensor distances,
    torch::Tensor indices,
    float sigma,
    float radius);


torch::Tensor compute_bin_stats(
    torch::Tensor bin_ids,
    torch::Tensor p_adj,
    int D);


torch::Tensor compute_alpha(
    torch::Tensor idx_map,
    torch::Tensor pix_ptrs,
    torch::Tensor raw,
    int N_loc
);


std::vector<torch::Tensor> compute_sparse_map(
    torch::Tensor means2D,
    torch::Tensor radii,
    int H,
    int W);

std::vector<torch::Tensor> merge_gaussians_cuda(
    torch::Tensor xyz,
    torch::Tensor feats,
    torch::Tensor scaling,
    torch::Tensor rotation,
    torch::Tensor opacity,
    torch::Tensor adjusted,
    torch::Tensor creation_steps,
    torch::Tensor idx_mat,
    torch::Tensor dist_mat,
    torch::Tensor low_mask,
    torch::Tensor keep_mask,
    float merge_dist);

#endif // EXIST_PROB_PRUNER_H