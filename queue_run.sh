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
#SBATCH -t 00:10:00
cd /scratch/bchk/jfahy/final_project # move to project dir
module load anaconda3_gpu/23.9.0
module load cudnn/8.9.0.131
module load python/3.10.13
eval "$(conda shell.bash hook)"      # initialize conda in env
conda activate unsloth_env           # activate conda env
python -m grpo.grpo --config config1 # launch grpo run