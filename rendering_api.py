import os
import subprocess


images_path = r"/media/gpu/input"
model_path = r"/tmp/"

dataset_folder_path = os.path.dirname(images_path)

# rendering
command = f'python render.py -s {dataset_folder_path} --model_path {model_path} --distributed_load'

subprocess.run(command, shell=True)