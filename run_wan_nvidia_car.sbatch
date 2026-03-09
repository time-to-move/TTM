#!/bin/bash
#SBATCH --job-name=run_wan_nvidia_car_t=3
#SBATCH --output=run_wan_nvidia_car_t=3.log
#SBATCH --error=run_wan_nvidia_car_t=3.log
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8

nvidia-smi

eval "$(micromamba shell hook -s bash)"
micromamba activate ----YOUR-ENV---

python run_wan.py \
  --input-path "./examples/nvidia_car_demo" \
  --output-path "./outputs/wan_nvidia_car_demo_sdedit_t=3.mp4" \
  --tweak-index 3 \
  --tstrong-index 3