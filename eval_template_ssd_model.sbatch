#!/bin/bash
#SBATCH --job-name=lm
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --constraint=APPLY_IF_NECESSARY
#SBATCH --time=THE_LONGER_THE_BETTER
#SBATCH --chdir=/private/home/xhan77/ssd-lm

source /private/home/xhan77/.bashrc
conda activate ssdlm

srun bash slurm_run_ssd_model_decode.sh 2022 "0,1,2,3,4,5,6,7" 1e-4 $TSTEP "n/a" "n/a" 5 25 1.0 ssd_dbs25 $LCTX $RTRUNC $DDEPTH $CTRLR $TOPP ssd_gen.jsonl
