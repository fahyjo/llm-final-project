#!/bin/bash
#SBATCH --job-name="tsp-grpo"
#SBATCH --output="tsp-grpo.%j.%N.out"
#SBATCH -e slurm-%j.err
#SBATCH --partition=gpuA100x4
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bchk-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 18:00:00
cd /scratch/bchk/jfahy/final_env     # move to project dir
eval "$(conda shell.bash hook)"      # initialize conda in env
conda activate unsloth_env           # activate conda env
python -m grpo.grpo --config config1 # launch grpo run