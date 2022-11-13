#!/bin/bash
#SBATCH --job-name=trainlm
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --constraint=APPLY_IF_NECESSARY
#SBATCH --time=THE_LONGER_THE_BETTER
#SBATCH --chdir=/private/home/xhan77/ssd-lm

source /private/home/xhan77/.bashrc
conda activate ssdlm

srun bash slurm_run_ssd_model_train.sh 2022 "0,1,2,3,4,5,6,7" 1e-4 5000 "xe" "no_dir" 5 25 "fp16" 1.0 "resume"
