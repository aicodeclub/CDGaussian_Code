#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# Ensure NCCL include directory is added (from Conda environment)
conda_prefix = os.environ.get("CONDA_PREFIX")
nccl_include = os.path.join(conda_prefix, "include") if conda_prefix else None
nccl_lib = os.path.join(conda_prefix, "lib") if conda_prefix else None

extra_include_dirs = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]
if nccl_include:
    extra_include_dirs.append(nccl_include)

extra_library_dirs = []
if nccl_lib:
    extra_library_dirs.append(nccl_lib)

setup(
    name="diff_gaussian_rasterization",
    packages=["diff_gaussian_rasterization"],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "cuda_rasterizer/timers.cu",
                "rasterize_points.cu",
                "exist_prob_pruner.cu",
                "ext.cpp"],
            include_dirs=extra_include_dirs,
            library_dirs=extra_library_dirs,
            libraries=["nccl"],
            extra_compile_args={
                "nvcc": [
                    "-std=c++17", "-arch=sm_89", "--extended-lambda", "--expt-relaxed-constexpr"
                ]
            }
        ),
        
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
