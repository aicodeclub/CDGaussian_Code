# scene/exist_prob_pruner.py
import torch
import torch.distributed as dist
import diff_gaussian_rasterization
from torch.cuda.amp import autocast
from scene.gaussian_model import GaussianModel
import utils.general_utils as utils
import numpy as np


def log_memory_usage(step_name, prune_idx):
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    return allocated


def eps():
    return 1e-6


class ExistProbPruner:
    def __init__(self,
                 prune_interval: int = 500,
                 sync_interval: int = 500,
                 warmup_steps: int = 2000,
                 max_gaussians: int = 4000000,
                 M_default: int = 50,
                 lambda_c: float = 0.0,
                 lambda_b: float = 0.0,
                 D_bins: int = 32,
                 x_min: float = -10.0,
                 x_max: float = 10.0,
                 sh_thr: float = 0.01,
                 view_alpha: float = 0.2,
                 #merge_dist: float = 0.1,
                 merge_dist: float = 0.5,
                 prune_passes: int = 1,
                 psnr_tol: float = 0.01,
                 prune_base: float = 0.15):
        self.prune_interval = prune_interval
        self.sync_interval = sync_interval
        self.warmup_steps = warmup_steps
        self.max_gaussians = max_gaussians
        self.M_default = M_default
        self.lambda_c = lambda_c
        self.lambda_b = lambda_b
        self.D = D_bins
        self.x_min = x_min
        self.x_max = x_max
        self.bin_size = (x_max - x_min) / D_bins
        self.sh_thr = sh_thr
        self.view_alpha = view_alpha
        self.merge_dist = merge_dist
        self.prune_passes = prune_passes
        self.psnr_tol = psnr_tol
        self.step_count = 0
        self.prob_threshold = 0.1
        self.existence_probs = None
        self.creation_steps = None
        self.global_bin_p = None
        self.global_bin_var = None
        self.scene_var = None
        self.prev_psnr = None
        self.centroids = None
        self.alpha_normalization_factor = 1.0
        self.prune_base = prune_base
        self._knn_cache = None
        self._knn_cache_n = -1

    def initialize(self, model: GaussianModel):
        N = model.get_xyz.shape[0]
        dev = model.get_xyz.device
        if self.existence_probs is None:
            self.existence_probs = torch.zeros(N, device=dev)
        elif self.existence_probs.numel() != N:
            ep = torch.zeros(N, device=dev)
            old = min(self.existence_probs.numel(), N)
            ep[:old] = self.existence_probs[:old]
            self.existence_probs = ep
        if self.creation_steps is None:
            self.creation_steps = torch.zeros(N, dtype=torch.int, device=dev)
        elif self.creation_steps.numel() != N:
            cs = torch.zeros(N, dtype=torch.int, device=dev)
            old = min(self.creation_steps.numel(), N)
            cs[:old] = self.creation_steps[:old]
            self.creation_steps = cs
        self.scene_var = torch.var(model.get_xyz, dim=0).sum().item()

        if self.centroids is None:
            xyz = model.get_xyz
            x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
            y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
            z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()

            x_centers = torch.linspace(x_min, x_max, 3, device=dev)
            y_centers = torch.linspace(y_min, y_max, 3, device=dev)
            z_centers = torch.linspace(z_min, z_max, 3, device=dev)

            self.centroids = torch.cartesian_prod(x_centers, y_centers, z_centers)

    def _invalidate_knn_cache(self):
        self._knn_cache = None
        self._knn_cache_n = -1

    def step(self, model: GaussianModel, ss_pkg, strategies, iteration, ground_truth=None):
        self.step_count += 1
        print(f"rank:{utils.GLOBAL_RANK}, step:{self.step_count}, iteration:{iteration}")
        if self.step_count == 1:
            dev = model.get_xyz.device
            self.global_bin_p = torch.zeros(self.D, device=dev)
            self.global_bin_var = torch.zeros(self.D, device=dev)

        if self.step_count <= self.warmup_steps:
            return
        with autocast():
            if self.step_count % self.prune_interval == 0:
                if ground_truth is not None:
                    self.prev_psnr = self._compute_psnr(model, ss_pkg, ground_truth)
                for prune_idx in range(self.prune_passes):
                    self.initialize(model)

                    A = log_memory_usage("Before _compute_weights",prune_idx)
                    alpha, view_weight, sh_weight = self._compute_weights(model, ss_pkg, strategies)
                    B = log_memory_usage("After _compute_weights",prune_idx)
                    if B - A >= 2:
                        print(f"rank:{utils.GLOBAL_RANK},pruneid:{prune_idx}, _compute_weights use memory:{B-A} GB")


                    '''
                    if self.alpha_normalization_factor == 1.0:
                        self.alpha_normalization_factor = alpha.max().item() if alpha.max() > 0 else 1.0
                    else:
                        self.alpha_normalization_factor = 0.9 * self.alpha_normalization_factor + 0.1 * alpha.max().item()

                    if self.alpha_normalization_factor > 0:
                        alpha = alpha / self.alpha_normalization_factor
                    '''


                    A = log_memory_usage("Before _message_passing_cuda", prune_idx)
                    N = model.get_xyz.size(0)
                    if (self.existence_probs is None) or (self.existence_probs.numel() != N) or ((self.existence_probs.max() - self.existence_probs.min()).abs() < 1e-12):
                        self.existence_probs = alpha.detach().clamp_min(1e-4)
                    local_p = self._message_passing_cuda(model, alpha, view_weight, sh_weight)
                    B = log_memory_usage("After _message_passing_cuda", prune_idx)
                    if B - A >= 2:
                        print(f"rank:{utils.GLOBAL_RANK},pruneid:{prune_idx}, _message_passing_cuda use memory:{B-A} GB")

                    A = log_memory_usage("Before _adjust_probs", prune_idx)
                    adjusted = self._adjust_probs(local_p, model)
                    #print(f"rank:{utils.GLOBAL_RANK}, pruneid:{prune_idx}, adjusted range: [{adjusted.min().item():.4f}, {adjusted.max().item():.4f}]")
                    B = log_memory_usage("After _adjust_probs", prune_idx)
                    if B - A >= 2:
                        print(f"rank:{utils.GLOBAL_RANK},pruneid:{prune_idx}, _adjust_probs use memory:{B-A} GB")


                    A = log_memory_usage("Before _merge_gaussians", prune_idx)
                    adjusted, merge_keep = self._merge_gaussians(model, adjusted)
                    if merge_keep is not None:
                        adjusted = adjusted[merge_keep]
                        self.creation_steps = self.creation_steps[merge_keep]
                    B = log_memory_usage("After _merge_gaussians", prune_idx)
                    if B - A >= 2:
                        print(f"rank:{utils.GLOBAL_RANK},pruneid:{prune_idx}, _merge_gaussians use memory:{B-A} GB")


                    A = log_memory_usage("Before _prune_decision", prune_idx)
                    #n_before = model.get_xyz.size(0)
                    keep_mask, validFlag = self._prune_decision(model, adjusted)
                    if validFlag == True:
                        print(f"rank:{utils.GLOBAL_RANK}, pruneid:{prune_idx}, gs size after _prune_decision:{model.get_xyz.size(0)}")
                    else:
                        continue
                    B = log_memory_usage("After _prune_decision", prune_idx)
                    if B - A >= 2:
                        print(f"rank:{utils.GLOBAL_RANK},pruneid:{prune_idx}, _prune_decision use memory:{B-A} GB")

                    '''
                    if keep_mask.sum().item() == n_before:
                        print(f"rank:{utils.GLOBAL_RANK}, pruneid:{prune_idx}, No points pruned in this pass, skipping")
                        break
                    '''

                    adjusted = adjusted[keep_mask]
                    if ground_truth is not None:
                        cur_psnr = self._compute_psnr(model, ss_pkg, ground_truth)
                        if self.prev_psnr - cur_psnr > self.psnr_tol:
                            model.restore_points(keep_mask)
                            break
                        self.prev_psnr = cur_psnr

                    A = log_memory_usage("Before adjusted.clone", prune_idx)
                    self.existence_probs = adjusted.clone()
                    if keep_mask is not None:
                        self.creation_steps = self.creation_steps[keep_mask].clone()
                    B = log_memory_usage("After adjusted.clone", prune_idx)
                    if B - A >= 2:
                        print(f"rank:{utils.GLOBAL_RANK},pruneid:{prune_idx}, adjusted.clone use memory:{B-A} GB")

            '''
            if (utils.DEFAULT_GROUP.size() > 1) and (self.step_count % self.sync_interval == 0):
                print(f"rank:{utils.GLOBAL_RANK},_sync_bins")
                self._sync_bins(model)
            '''

        torch.cuda.empty_cache()

    def _compute_psnr(self, model, ss_pkg, gt):
        rendered = model.render(ss_pkg)
        mse = torch.mean((rendered - gt) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


    def _compute_weights(self, model, ss_pkg, strategies):
        dev = model.get_xyz.device
        N = model.get_xyz.size(0)

        if N == 0:
            zero = torch.zeros(0, device=dev)
            return zero, zero, zero

        alpha = torch.zeros(N, device=dev)
        V = 0

        for cam_id, strat in enumerate(strategies):
            if utils.GLOBAL_RANK not in strat.gpu_ids:
                continue

            raw = ss_pkg["batched_conic_opacity_redistributed"][cam_id]
            if raw.numel() == 0:
                continue
            raw = raw.float().clamp(min=0.0)
            if raw.dim() > 1:
                raw = raw.mean(dim=-1)

            m2d = ss_pkg["batched_means2D_redistributed"][cam_id].contiguous().float()
            radii = ss_pkg["batched_radii_redistributed"][cam_id].contiguous().float()
            Nloc = m2d.size(0)
            if Nloc == 0:
                continue
            V += 1

            idx_map, ptrs = diff_gaussian_rasterization._C.compute_sparse_map(m2d, radii, *utils.get_img_size())
            if idx_map.numel() == 0:
                continue

            alpha_v = diff_gaussian_rasterization._C.compute_alpha(idx_map, ptrs, raw, Nloc)
            if alpha_v.numel() > 0:
                alpha_v = alpha_v.clamp(min=0.0)
                length = min(N, alpha_v.numel())
                alpha[:length] += alpha_v[:length]

        if V > 0:
            alpha = (alpha / float(V)).clamp_(0.0, 1.0)

        view_weight = alpha.clone()

        sh_coeffs = model.get_features
        if sh_coeffs is None or sh_coeffs.numel() == 0:
            sh_weight = torch.zeros(N, device=dev)
        else:
            sh_coeffs = sh_coeffs.to(dev)
            sh_norm = torch.linalg.vector_norm(sh_coeffs, ord=2, dim=1)
            if sh_norm.numel() == 0:
                sh_weight = torch.zeros(N, device=dev)
            else:
                max_sh_norm = torch.max(sh_norm).clamp(min=eps())
                sh_weight = sh_norm / max_sh_norm

        return alpha, view_weight, sh_weight

    def _message_passing_cuda(self, model, alpha, view_weight, sh_weight):
        #dev = model.get_xyz.device
        xyz = model.get_xyz.contiguous()
        #N = xyz.size(0)


        #M_raw = self._adaptive_M(xyz)
        #M = min(M_raw, self.M_default)
        #sigma = max(0.05, min(0.2, self._median_dist(xyz) / 2))
        sigma = 0.1

        #age_w = torch.clamp(1 - (self.step_count - self.creation_steps.float()) / 2000, 0.1, 1.0)
        age_w = torch.clamp(1 - (self.step_count - self.creation_steps.float()) / 1000, 0.1, 1.0)

        dist_mat, idx_mat = diff_gaussian_rasterization._C.compute_distances(xyz, self.M_default, self.merge_dist)
        local_p = diff_gaussian_rasterization._C.compute_local_probs(
            self.existence_probs,
            alpha,
            age_w,
            view_weight,
            sh_weight,
            dist_mat,
            idx_mat,
            sigma,
            self.merge_dist
        )

        self._knn_cache = (idx_mat, dist_mat)
        self._knn_cache_n = xyz.shape[0]
        return local_p

    def _adjust_probs(self, local_p, model):
        N = model.get_xyz.size(0)

        if self.creation_steps.numel() != N:
            cs = torch.zeros(N, dtype=self.creation_steps.dtype, device=self.creation_steps.device)
            old = self.creation_steps.numel()
            cs[:old] = self.creation_steps
            self.creation_steps = cs

        if self.existence_probs.numel() != N:
            ep = torch.zeros(N, device=self.existence_probs.device)
            old = self.existence_probs.numel()
            ep[:old] = self.existence_probs
            self.existence_probs = ep

        age_w = torch.clamp(1 - (self.step_count - self.creation_steps.float()) / 1000, min=0.1, max=1.0)

        xyz = model.get_xyz
        if self.centroids is None:
            self.initialize(model)

        dists = torch.cdist(xyz, self.centroids)
        min_dists = torch.min(dists, dim=1)[0]

        boundary_weight = 1.0 / (1.0 + min_dists)
        boundary_weight = boundary_weight / boundary_weight.max()

        adjusted = local_p[:N] + self.lambda_c * age_w + self.lambda_b * boundary_weight

        adjusted = torch.clamp(adjusted, 0.0, 1.0)
        return adjusted


    def _merge_gaussians(self, model, adjusted):
        with torch.no_grad():
            low_mask = (adjusted < 0.05)  # bool
            n_low = int(low_mask.sum().item())
            if n_low < 2:
                return adjusted, None

            xyz      = model.get_xyz.contiguous()
            feats    = model.get_features.contiguous()
            scaling  = model.get_scaling.contiguous()
            rotation = model.get_rotation.contiguous()
            opacity  = model.get_opacity.contiguous()

            keep_mask = torch.ones_like(low_mask, dtype=torch.int32, device=low_mask.device)

            use_cache = (
                hasattr(self, "_knn_cache") and
                self._knn_cache is not None and
                getattr(self, "_knn_cache_n", -1) == xyz.shape[0]
            )
            if use_cache:
                idx_mat, dist_mat = self._knn_cache
            else:
                dist_mat, idx_mat = diff_gaussian_rasterization._C.compute_distances(
                    xyz, int(self.M_default), float(self.merge_dist)
                )

            dist_mat = dist_mat.to(device=xyz.device)
            idx_mat  = idx_mat.to(device=xyz.device, dtype=torch.long)


            MAX_LOW = 200_000
            if n_low > MAX_LOW:
                cand_idx = torch.nonzero(low_mask, as_tuple=False).squeeze(1)
                vox = torch.floor(xyz[cand_idx] / max(self.merge_dist, 1e-4)).long()
                

                uniq_vox, inverse = torch.unique(vox, dim=0, return_inverse=True)

                order = torch.argsort(inverse)
                inv_sorted = inverse[order]
                first_mask = torch.ones_like(inv_sorted, dtype=torch.bool)
                first_mask[1:] = inv_sorted[1:] != inv_sorted[:-1]
                first_pos = order[first_mask]

                picked = cand_idx[first_pos]

                if picked.numel() > MAX_LOW:
                    picked = picked[:MAX_LOW]
                new_low = torch.zeros_like(low_mask)
                new_low[picked] = True
                low_mask = new_low
                n_low = int(low_mask.sum().item())

            (new_keep, new_xyz, new_feats, new_scaling, new_rotation,
             new_opacity, new_p, new_steps) = diff_gaussian_rasterization._C.merge_gaussians_cuda(
                xyz, feats, scaling, rotation, opacity,
                adjusted.contiguous(),
                self.creation_steps.contiguous(),
                idx_mat, dist_mat,
                low_mask.contiguous(),
                keep_mask.contiguous(),  # int32
                float(self.merge_dist)
            )

            if new_keep.all().item():
                return adjusted, None

            merged_in = int((~new_keep).sum().item())
            created   = int(new_xyz.size(0))
            print(f"Merged {merged_in} old points into {created} new points")

            model.prune_points((~new_keep).to(dtype=torch.bool))
            model.add_points(new_xyz, new_p, new_feats, new_scaling, new_rotation, new_opacity)


            newN = int(new_keep.sum().item()) + int(new_p.numel())
            out_adj = torch.empty(newN, dtype=adjusted.dtype, device=adjusted.device)
            out_adj[:int(new_keep.sum().item())] = adjusted[new_keep]
            out_adj[int(new_keep.sum().item()):] = new_p
            adjusted = out_adj

            out_steps = torch.empty(newN, dtype=self.creation_steps.dtype, device=self.creation_steps.device)
            out_steps[:int(new_keep.sum().item())] = self.creation_steps[new_keep]
            out_steps[int(new_keep.sum().item()):] = new_steps
            self.creation_steps = out_steps

            keep_all = torch.ones(adjusted.shape[0], dtype=torch.bool, device=adjusted.device)
            return adjusted, keep_all



    def _prune_decision(self, model, adjusted):
        base = self.prune_base
        beta = 0.3

        scene_complexity = min(1.0, model.get_xyz.size(0) / self.max_gaussians)
        #complexity_factor = 0.5 + 0.5 * scene_complexity
        complexity_factor = 0.5 * scene_complexity

        theta_step = base + beta * complexity_factor * (1 - self.step_count / 30000)
        #theta_step = max(0.05, min(0.5, theta_step))
        #theta_step = base + beta * scene_complexity * (self.step_count / 30000)

        keep = adjusted > theta_step
        #print(f"rank:{utils.GLOBAL_RANK}, base:{base}, beta:{beta}, interval:{self.prune_interval}, warmup:{self.warmup_steps}, passes:{self.prune_passes}")
        #print(f"rank:{utils.GLOBAL_RANK}, threshold:{theta_step:.4f},keeping {keep.sum().item()}/{adjusted.size(0)} points")

        if int(keep.sum().item()) <= 0.65 * adjusted.size(0):
            print(f"rank:{utils.GLOBAL_RANK}, keep ratio {int(keep.sum().item())}/{adjusted.size(0):} <= 0.65, skip pruning this pass")
            self.prune_base = base * 0.1
            self.prune_interval = self.prune_interval + 200
            return keep, False

        if keep.sum() > self.max_gaussians:
            _, idx = torch.topk(adjusted, self.max_gaussians)
            keep = torch.zeros_like(keep)
            keep[idx] = True
            print(f"Limited to max_gaussians: {self.max_gaussians}")

        model.prune_points(~keep)
        return keep, True

    def _sync_bins(self, model):
        x = model.get_xyz[:, 0]
        bin_ids = ((x - self.x_min) / self.bin_size).floor().clamp(0, self.D - 1).long()

        stats = diff_gaussian_rasterization._C.compute_bin_stats(bin_ids, self.existence_probs, self.D)

        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=utils.MP_GROUP)

        sum_p, cnt_f, sum_sq = stats[0], stats[1], stats[2]
        eps_val = 1e-6
        self.global_bin_p = sum_p / (cnt_f + eps_val)
        self.global_bin_var = sum_sq / (cnt_f + eps_val) - self.global_bin_p * self.global_bin_p
        self.has_sync = True

    def _adaptive_M(self, xyz):
        if xyz.size(0) == 0:
            return self.M_default
        dists, _ = diff_gaussian_rasterization._C.compute_distances(xyz, 10, 0.5)
        avg_d = dists.mean(dim=1)
        dens = 1 / (avg_d + eps())
        M_tensor = (dens / dens.max() * self.M_default).int().clamp(min=10)
        return int(M_tensor.max().item())

    def _median_dist(self, xyz):
        if xyz.size(0) == 0:
            return self.merge_dist
        dists, _ = diff_gaussian_rasterization._C.compute_distances(xyz, 10, 0.5)
        return float(torch.median(dists))
