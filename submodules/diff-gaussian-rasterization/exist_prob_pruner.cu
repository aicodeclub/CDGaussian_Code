// diff_gaussian_rasterization/exist_prob_pruner_cuda.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <algorithm>
#include <thrust/sort.h>
#include <ATen/ATen.h>


const int BLOCK_SIZE = 256;
const int MAX_K = 50;
const int MAX_PIXELS_PER_GAUSSIAN = 50;

__global__ void optimized_knn_kernel(
    const float* __restrict__ pts,
    const int num_points,
    const int k,
    const float max_radius_sq,
    float* __restrict__ dists,
    int* __restrict__ indices)
{

    __shared__ float3 block_pts[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float3 query = make_float3(pts[idx*3], pts[idx*3+1], pts[idx*3+2]);


    float min_dists[MAX_K];
    int min_indices[MAX_K];
    for (int i=0; i<k; i++) {
        min_dists[i] = max_radius_sq;
        min_indices[i] = -1;
    }


    for (int block_start=0; block_start<num_points; block_start+=BLOCK_SIZE) {
        int load_idx = block_start + threadIdx.x;
        float3 loaded_point = make_float3(0.0f, 0.0f, 0.0f);
        if (load_idx < num_points) {
            loaded_point = make_float3(
                pts[load_idx*3],
                pts[load_idx*3+1],
                pts[load_idx*3+2]);
        }
        block_pts[threadIdx.x] = loaded_point;
        __syncthreads();

        int valid_points = min(BLOCK_SIZE, num_points - block_start);
        for (int i=0; i<valid_points; i++) {
            if (block_start + i == idx || block_pts[i].x == 0.0f) continue;
            float3 p = block_pts[i];
            float dx = query.x - p.x;
            float dy = query.y - p.y;
            float dz = query.z - p.z;
            float dist_sq = dx*dx + dy*dy + dz*dz;

            if (dist_sq < min_dists[k-1]) {
                int pos = k-1;
                while (pos > 0 && dist_sq < min_dists[pos-1]) {
                    min_dists[pos] = min_dists[pos-1];
                    min_indices[pos] = min_indices[pos-1];
                    pos--;
                }
                if (pos >= 0 && pos < k) {
                    min_dists[pos] = dist_sq;
                    min_indices[pos] = block_start + i;
                }
            }
        }
        __syncthreads();
    }

    for (int i=0; i<k; i++) {
        if (idx < num_points && i < k) {
            dists[idx*k + i] = sqrtf(min_dists[i]);
            indices[idx*k + i] = min_indices[i];
        }
    }
}


std::vector<torch::Tensor> compute_distances(
    torch::Tensor points,
    int k,
    float max_radius)
{
    TORCH_CHECK(points.dim() == 2, "Points must be 2D tensor");
    TORCH_CHECK(points.size(1) == 3, "Points must have 3 coordinates");
    TORCH_CHECK(k > 0 && k <= MAX_K, "k must be between 1 and MAX_K");

    int N = points.size(0);
    auto dists = torch::empty({N, k}, points.options());
    auto indices = torch::empty({N, k}, torch::kInt32).to(points.device());

    TORCH_CHECK(points.is_cuda(), "Input must be on GPU");
    TORCH_CHECK(dists.is_cuda(), "Output dists must on GPU");
    TORCH_CHECK(indices.is_cuda(), "Output indices must on GPU");

    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    optimized_knn_kernel<<<blocks, BLOCK_SIZE>>>(
        points.data_ptr<float>(),
        N,
        k,
        max_radius * max_radius,
        dists.data_ptr<float>(),
        indices.data_ptr<int>()
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    return {dists, indices};
}

__global__ void compute_local_probs_kernel(
    const float* __restrict__ existence_probs,
    const float* __restrict__ alpha,
    const float* __restrict__ age_weight,
    const float* __restrict__ view_weight,
    const float* __restrict__ sh_weight,
    const float* __restrict__ distances,
    const int* __restrict__ indices,
    float* __restrict__ local_probs,
    int N, int M, float sigma, float radius
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float* dist_i = distances + i * M;
    const int* idx_i = indices + i * M;
    float sum_w = 0.0f, sum_wp = 0.0f;
    for (int k = 0; k < M; ++k) {
        int j = idx_i[k];
        float d = dist_i[k];
        if (d >= radius || j < 0 || j >= N) continue;
        float w = expf(-d / sigma)
                  * alpha[j]
                  * view_weight[j]
                  * sh_weight[j];
        sum_w  += w;
        sum_wp += w * existence_probs[j];
    }
    local_probs[i] = (sum_w > 0.0f ? sum_wp / sum_w : 0.0f);
}

torch::Tensor compute_local_probs(
    torch::Tensor existence_probs,
    torch::Tensor alpha,
    torch::Tensor age_weight,
    torch::Tensor view_weight,
    torch::Tensor sh_weight,
    torch::Tensor distances,
    torch::Tensor indices,
    float sigma,
    float radius
)
{
    auto N = existence_probs.size(0);
    auto M = distances.size(1);
    auto local_probs = torch::zeros({N}, existence_probs.options());
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    compute_local_probs_kernel<<<blocks, threads>>>(
        existence_probs.data_ptr<float>(),
        alpha.data_ptr<float>(),
        age_weight.data_ptr<float>(),
        view_weight.data_ptr<float>(),
        sh_weight.data_ptr<float>(),
        distances.data_ptr<float>(),
        indices.data_ptr<int>(),
        local_probs.data_ptr<float>(),
        N, M, sigma, radius
    );
    return local_probs;
}


__global__ void compute_bin_stats_kernel(
    const int* __restrict__ bin_ids,
    const float* __restrict__ p_adj,
    float* __restrict__ sum_p,
    int* __restrict__ cnt,
    float* __restrict__ sum_sq,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int b = bin_ids[idx];
    float v = p_adj[idx];

    atomicAdd(&sum_p[b], v);
    atomicAdd(&sum_sq[b], v * v);
    atomicAdd(&cnt[b], 1);
}

torch::Tensor compute_bin_stats(
    torch::Tensor bin_ids,
    torch::Tensor p_adj,
    int D)
{

    bin_ids = bin_ids.contiguous().to(torch::kInt32);
    p_adj   = p_adj.contiguous().to(torch::kFloat32);
    int N = bin_ids.size(0);


    auto options_f = torch::TensorOptions().dtype(torch::kFloat32)
                                          .device(p_adj.device());
    auto options_i = torch::TensorOptions().dtype(torch::kInt32)
                                          .device(p_adj.device());
    auto sum_p  = torch::zeros({D}, options_f);
    auto cnt    = torch::zeros({D}, options_i);
    auto sum_sq = torch::zeros({D}, options_f);


    const int threads = 512;
    const int blocks  = (N + threads - 1) / threads;
    compute_bin_stats_kernel<<<blocks, threads>>>(
        bin_ids.data_ptr<int>(),
        p_adj.data_ptr<float>(),
        sum_p.data_ptr<float>(),
        cnt.data_ptr<int>(),
        sum_sq.data_ptr<float>(),
        N
    );

    cudaDeviceSynchronize();


    auto cnt_f = cnt.to(torch::kFloat32);
    auto stats = torch::empty({3, D}, options_f);
    // stats[0] ← sum_p, stats[1] ← cnt_f, stats[2] ← sum_sq
    stats[0].copy_(sum_p);
    stats[1].copy_(cnt_f);
    stats[2].copy_(sum_sq);

    return stats;
}


__global__ void compute_alpha_kernel(
    const int* __restrict__ idx_map,
    const int* __restrict__ pix_ptrs,
    const float* __restrict__ raw,
    float* __restrict__ alpha_accum,
    int P,
    int N
)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;

    int start = pix_ptrs[p];
    int end = pix_ptrs[p+1];
    int total = pix_ptrs[P];
    if (start < 0 || start > total || end < 0 || end > total) return;
    if (start == end) return;

    float T_before = 1.0f;
    for (int idx = start; idx < end; ++idx) {
        int j = idx_map[idx];
        if (j < 0 || j >= N) continue;
        float o_j = raw[j];
        float contrib = o_j * T_before;
        atomicAdd(&alpha_accum[j], contrib);
        T_before *= (1.0f - o_j);
    }
}

torch::Tensor compute_alpha(
    torch::Tensor idx_map,
    torch::Tensor pix_ptrs,
    torch::Tensor raw,
    int N_loc
)
{
    auto P = pix_ptrs.numel() - 1;
    auto alpha_accum = torch::zeros({N_loc}, raw.options());
    const int threads = 256;
    const int blocks = (P + threads - 1) / threads;
    compute_alpha_kernel<<<blocks, threads>>>(
        idx_map.data_ptr<int>(),
        pix_ptrs.data_ptr<int>(),
        raw.data_ptr<float>(),
        alpha_accum.data_ptr<float>(),
        P,
        N_loc
    );
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error: ") + cudaGetErrorString(err));
    }
    return alpha_accum;
}


__global__ void count_coverage_kernel(
    const float* __restrict__ means2D,  // [N_loc*2]
    const float* __restrict__ radii,    // [N_loc]
    int*    __restrict__ pixel_count,
    int N_loc, int H, int W
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N_loc) return;

    float cx = means2D[2*j + 0];
    float cy = means2D[2*j + 1];
    float r  = radii[j];
    int   rpx = ceilf(r);
    int xmin = max(0, int(floorf(cx - r)));
    int xmax = min(W-1, int( ceilf(cx + r)));
    int ymin = max(0, int(floorf(cy - r)));
    int ymax = min(H-1, int( ceilf(cy + r)));

    float r2 = r * r;
    for (int y = ymin; y <= ymax; ++y) {
        for (int x = xmin; x <= xmax; ++x) {
            float dx = x + 0.5f - cx;
            float dy = y + 0.5f - cy;
            if (dx*dx + dy*dy <= r2) {
                int p = y * W + x;
                atomicAdd(&pixel_count[p], 1);
            }
        }
    }
}


__global__ void fill_indices_kernel_optimized(
    const float* __restrict__ means2D,
    const float* __restrict__ radii,
    const int* __restrict__ pix_ptrs,
    int*    __restrict__ write_ptrs,
    int*    __restrict__ idx_map,
    int N_loc, int H, int W
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N_loc) return;

    float cx = means2D[2*j + 0];
    float cy = means2D[2*j + 1];
    float r = radii[j];
    int r_px = min(static_cast<int>(ceilf(r)), 10);

    int xmin = max(0, int(floorf(cx - r_px)));
    int xmax = min(W-1, int(ceilf(cx + r_px)));
    int ymin = max(0, int(floorf(cy - r_px)));
    int ymax = min(H-1, int(ceilf(cy + r_px)));

    float r2 = r * r;
    for (int y = ymin; y <= ymax; ++y) {
        for (int x = xmin; x <= xmax; ++x) {
            float dx = x + 0.5f - cx;
            float dy = y + 0.5f - cy;
            if (dx*dx + dy*dy <= r2) {
                int p = y * W + x;
                int max_allowed = pix_ptrs[p+1] - pix_ptrs[p];
                if (max_allowed <= 0) continue;

                int pos = atomicAdd(&write_ptrs[p], 1);
                if (pos < max_allowed) {
                    idx_map[pix_ptrs[p] + pos] = j;
                }
            }
        }
    }
}

std::vector<torch::Tensor> compute_sparse_map(
    torch::Tensor means2D,
    torch::Tensor radii,
    int H,
    int W
)
{
    auto N_loc = means2D.size(0);
    int P = H * W;
    auto pixel_count = torch::zeros({P}, means2D.options().dtype(torch::kInt32));
    const int threads = 256;
    int blocks = (N_loc + threads - 1) / threads;

    count_coverage_kernel<<<blocks, threads>>>(
        means2D.data_ptr<float>(),
        radii.data_ptr<float>(),
        pixel_count.data_ptr<int>(),
        N_loc, H, W
    );

    auto pix_ptrs = torch::empty({P+1}, pixel_count.options().dtype(torch::kInt32));
    {
        thrust::device_ptr<int> d_count(pixel_count.data_ptr<int>());
        thrust::device_ptr<int> d_ptrs(pix_ptrs.data_ptr<int>());
        thrust::transform(thrust::device,
                         d_count, d_count + P, d_count,
                         [] __device__ (int x) {
                             return x < MAX_PIXELS_PER_GAUSSIAN ? x : MAX_PIXELS_PER_GAUSSIAN;
                         });
        thrust::exclusive_scan(d_count, d_count + P, d_ptrs + 1);
        cudaMemset(d_ptrs.get(), 0, sizeof(int));
    }

    int total = 0;
    cudaMemcpy(&total, pix_ptrs.data_ptr<int>() + P, sizeof(int), cudaMemcpyDeviceToHost);
    auto idx_map = torch::empty({total}, torch::dtype(torch::kInt32).device(means2D.device()));
    auto write_ptrs = torch::zeros_like(pix_ptrs.slice(0, 0, P));

    fill_indices_kernel_optimized<<<blocks, threads>>>(
        means2D.data_ptr<float>(),
        radii.data_ptr<float>(),
        pix_ptrs.data_ptr<int>(),
        write_ptrs.data_ptr<int>(),
        idx_map.data_ptr<int>(),
        N_loc, H, W
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error: ") + cudaGetErrorString(err));
    }
    return {idx_map, pix_ptrs};
}


template <typename scalar_t, typename index_t>
__global__ void merge_gaussians_kernel(
    const scalar_t* __restrict__ xyz,
    const scalar_t* __restrict__ feats,
    const scalar_t* __restrict__ scaling,
    const scalar_t* __restrict__ rotation,
    const scalar_t* __restrict__ opacity,
    const scalar_t* __restrict__ adjusted,
    const int*      __restrict__ creation_steps,
    const index_t*  __restrict__ idx_mat,
    const scalar_t* __restrict__ dist_mat,
    const bool*     __restrict__ low_mask,
    int*            __restrict__ keep_mask,   // <<== int32
    int N, int M, int F, int Ss, int Sr,
    scalar_t merge_dist,
    int*          __restrict__ out_count,
    scalar_t*     __restrict__ out_xyz,
    scalar_t*     __restrict__ out_feats,
    scalar_t*     __restrict__ out_scaling,
    scalar_t*     __restrict__ out_rotation,
    scalar_t*     __restrict__ out_opacity,
    scalar_t*     __restrict__ out_probs,
    int*          __restrict__ out_steps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (!low_mask[i]) return;
    if (atomicCAS(&keep_mask[i], 1, 1) == 0) return;

    scalar_t xi = xyz[3*i], yi = xyz[3*i+1], zi = xyz[3*i+2];

    for (int k = 0; k < M; ++k) {
        int j = idx_mat[i*M + k];
        if (j <= i) continue;
        if (!low_mask[j]) continue;
        if (atomicCAS(&keep_mask[j], 1, 1) == 0) continue;
        scalar_t d = dist_mat[i*M + k];
        if (d >= merge_dist) { continue; }

        scalar_t dot=0, ni2=0, nj2=0;
        for (int f = 0; f < F; ++f) {
            scalar_t a = feats[i*F+f], b = feats[j*F+f];
            dot += a*b; ni2 += a*a; nj2 += b*b;
        }
        scalar_t sim = dot / (sqrt(ni2)*sqrt(nj2) + 1e-8);
        if (sim <= (scalar_t)0.9) continue;

        if (atomicCAS(&keep_mask[i], 1, 0) != 1) {
            atomicExch(&keep_mask[j], 1);
            continue;
        }

        atomicExch(&keep_mask[j], 0);

        int idx = atomicAdd(out_count, 1);

        out_xyz[3*idx+0] = (scalar_t)0.5*(xi + xyz[3*j+0]);
        out_xyz[3*idx+1] = (scalar_t)0.5*(yi + xyz[3*j+1]);
        out_xyz[3*idx+2] = (scalar_t)0.5*(zi + xyz[3*j+2]);

        for (int f = 0; f < F; ++f)
            out_feats[idx*F+f] = (scalar_t)0.5*(feats[i*F+f] + feats[j*F+f]);

        for (int s = 0; s < Ss; ++s)
            out_scaling[idx*Ss+s] = (scalar_t)0.5*(scaling[i*Ss+s] + scaling[j*Ss+s]);

        for (int r = 0; r < Sr; ++r)
            out_rotation[idx*Sr+r] = (scalar_t)0.5*(rotation[i*Sr+r] + rotation[j*Sr+r]);

        if (Sr == 4) {
            scalar_t q0 = out_rotation[idx*Sr+0];
            scalar_t q1 = out_rotation[idx*Sr+1];
            scalar_t q2 = out_rotation[idx*Sr+2];
            scalar_t q3 = out_rotation[idx*Sr+3];
            scalar_t nn = sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3) + 1e-12;
            out_rotation[idx*Sr+0] = q0/nn;
            out_rotation[idx*Sr+1] = q1/nn;
            out_rotation[idx*Sr+2] = q2/nn;
            out_rotation[idx*Sr+3] = q3/nn;
        }

        out_opacity[idx] = (scalar_t)0.5*(opacity[i] + opacity[j]);
        out_probs[idx]   = fmax(adjusted[i], adjusted[j]);
        out_steps[idx]   = max(creation_steps[i], creation_steps[j]);

        break;
    }
}


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
    float merge_dist
)
{
    int N = xyz.size(0), M = idx_mat.size(1), F = feats.size(1);
    int Ss = scaling.size(1), Sr = rotation.size(1);
    auto opts = xyz.options();
    auto out_xyz      = torch::empty({N,3}, opts);
    auto out_feats    = torch::empty({N,F}, opts);
    auto out_scaling  = torch::empty({N,Ss}, opts);
    auto out_rotation = torch::empty({N,Sr}, opts);
    auto out_opacity  = torch::empty({N,1}, opts);
    auto out_probs    = torch::empty({N}, opts);
    auto out_steps    = torch::empty({N}, torch::dtype(torch::kInt32).device(xyz.device()));
    auto out_count    = torch::zeros({1}, torch::dtype(torch::kInt32).device(xyz.device()));
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    using scalar_t = float;

    TORCH_CHECK(xyz.is_cuda(), "Input must be on GPU");
    TORCH_CHECK(feats.is_cuda(), "Input must be on GPU");
    TORCH_CHECK(scaling.is_cuda(), "Input must be on GPU");
    TORCH_CHECK(rotation.is_cuda(), "Input must on GPU");
    TORCH_CHECK(opacity.is_cuda(), "Input must on GPU");
    TORCH_CHECK(adjusted.is_cuda(), "Input must on GPU");
    TORCH_CHECK(creation_steps.is_cuda(), "Input must on GPU");
    TORCH_CHECK(idx_mat.is_cuda(), "Input must on GPU");
    TORCH_CHECK(dist_mat.is_cuda(), "Input must on GPU");
    TORCH_CHECK(low_mask.is_cuda(), "Input must on GPU");
    TORCH_CHECK(keep_mask.is_cuda(), "Input must on GPU");

    xyz = xyz.contiguous();
    feats = feats.contiguous();
    scaling = scaling.contiguous();
    rotation = rotation.contiguous();
    opacity = opacity.contiguous();
    adjusted = adjusted.contiguous();
    creation_steps = creation_steps.contiguous();
    idx_mat = idx_mat.contiguous();
    dist_mat = dist_mat.contiguous();
    low_mask = low_mask.contiguous();
    keep_mask = keep_mask.contiguous();

    merge_gaussians_kernel<scalar_t,int64_t><<<blocks,threads>>>(
            xyz.data_ptr<scalar_t>(),
            feats.data_ptr<scalar_t>(),
            scaling.data_ptr<scalar_t>(),
            rotation.data_ptr<scalar_t>(),
            opacity.data_ptr<scalar_t>(),
            adjusted.data_ptr<scalar_t>(),
            creation_steps.data_ptr<int>(),
            idx_mat.data_ptr<int64_t>(),
            dist_mat.data_ptr<scalar_t>(),
            low_mask.data_ptr<bool>(),
            keep_mask.data_ptr<int>(),
            N, M, F, Ss, Sr,
            (scalar_t)merge_dist,
            out_count.data_ptr<int>(),
            out_xyz.data_ptr<scalar_t>(),
            out_feats.data_ptr<scalar_t>(),
            out_scaling.data_ptr<scalar_t>(),
            out_rotation.data_ptr<scalar_t>(),
            out_opacity.data_ptr<scalar_t>(),
            out_probs.data_ptr<scalar_t>(),
            out_steps.data_ptr<int>()
        );
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    int cnt = out_count.item<int>();
    auto keep_out = keep_mask.to(torch::kBool);
    auto merged_xyz      = out_xyz.narrow(0, 0, cnt);
    auto merged_feats    = out_feats.narrow(0, 0, cnt);
    auto merged_scaling  = out_scaling.narrow(0, 0, cnt);
    auto merged_rotation = out_rotation.narrow(0, 0, cnt);
    auto merged_opacity  = out_opacity.narrow(0, 0, cnt);
    auto merged_probs    = out_probs.narrow(0, 0, cnt);
    auto merged_steps    = out_steps.narrow(0, 0, cnt);

    return { keep_out,
             merged_xyz,
             merged_feats,
             merged_scaling,
             merged_rotation,
             merged_opacity,
             merged_probs,
             merged_steps };
}
