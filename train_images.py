import os
import subprocess
#os.environ['TORCH_USE_CUDA_DSA'] = "1"
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


images_path = r"/root/autodl-tmp/code/input"


folder_path = os.path.dirname(images_path)


#command = f'python convert.py -s {folder_path}'
#subprocess.run(command, shell=True)

# single GPU, batchSize = 1, non-distributed training
#command = f'/root/miniconda3/bin/python train.py -s {folder_path}'

# 1 GPU, batchSize = 1,non-distributed training
#command = f'torchrun --standalone --nnodes=1 --nproc-per-node=1 train.py --bsz 1 -s {folder_path}'

# 2 GPU, batchSize = 1,distributed training
#command = f'/root/miniconda3/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py --bsz 2 -s {folder_path}'

# 4 GPU, batchSize = 4,distributed training
command = f'/root/miniconda3/bin/torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --bsz 4 -s {folder_path}'

subprocess.run(command, shell=True)