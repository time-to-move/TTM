#!/bin/bash

# Iterate over video names
for t in {2..6}; do
    # Create a temporary batch script
    job_name="run_wan_nvidia_car_t=${t}"
    batch_script=$(mktemp)
    cat <<EOT > "$batch_script"
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=${job_name}.log
#SBATCH --error=${job_name}.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1

echo "==== Job: $job_name ===="
echo "Started: \$(date -Is) on \$(hostname)"
nvidia-smi
pwd

eval "\$(micromamba shell hook -s bash)"
micromamba activate alltracker

python run_wan.py \
  --input-path "./examples/nvidia_car_demo" \
  --output-path "./outputs/wan_nvidia_car_demo_sdedit_t=${t}.mp4" \
  --tweak-index $t \
  --tstrong-index $t

EOT

  # Output the script for inspection
  cat $batch_script
  # Submit it
  # sbatch $batch_script
  # Remove it
  rm $batch_script
done