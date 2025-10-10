/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"
#include "exist_prob_pruner.h"
#include "cuda_rasterizer/config.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mark_visible", &markVisible);
  m.def("preprocess_gaussians", &PreprocessGaussiansCUDA);
  m.def("preprocess_gaussians_backward", &PreprocessGaussiansBackwardCUDA);
  m.def("get_distribution_strategy", &GetDistributionStrategyCUDA);
  m.def("render_gaussians", &RenderGaussiansCUDA);
  m.def("render_gaussians_backward", &RenderGaussiansBackwardCUDA);
  m.def("get_local2j_ids_bool", &GetLocal2jIdsBoolCUDA);
  m.def("get_topk_idx", &get_topk_idx);
  m.def("get_local2j_ids_bool_adjust_mode6", &GetLocal2jIdsBoolAdjustMode6CUDA);

  // Image Distribution Utilities
  m.def("get_touched_locally", &GetTouchedLocally);
  m.def("load_image_tiles_by_pos", &LoadImageTilesByPos);
  m.def("set_image_tiles_by_pos", &SetImageTilesByPos);
  m.def("get_pixels_compute_locally_and_in_rect", &GetPixelsComputeLocallyAndInRect);
  m.def("get_block_XY", &GetBlockXY);

  m.def("compute_distances", &compute_distances);
  m.def("compute_local_probs", &compute_local_probs,"compute_local_probs(existence_probs, opacity, transmittance, view_w, sh_w, distances, indices, sigma, radius)");
  m.def("compute_bin_stats", &compute_bin_stats,"compute_bin_stats(bin_ids, p_adj, D) -> (sum_probs_local, count_local)");
  m.def("compute_alpha", &compute_alpha);
  m.def("compute_sparse_map", &compute_sparse_map);
  m.def("merge_gaussians_cuda", &merge_gaussians_cuda);
}