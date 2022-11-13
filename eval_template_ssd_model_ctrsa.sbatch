#!/bin/bash
#SBATCH --job-name=ctrlm
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=160G
#SBATCH --gpus-per-node=4
#SBATCH --constraint=APPLY_IF_NECESSARY
#SBATCH --time=THE_LONGER_THE_BETTER
#SBATCH --chdir=/private/home/xhan77/ssd-lm

source /private/home/xhan77/.bashrc
conda activate ssdlm

srun bash slurm_run_ssd_model_decode_fin.sh 2022 "n/a" 1e-4 $TSTEP "n/a" "fin|/private/home/xhan77/ssd-lm/logging/pplm_discrim_prompts.jsonl" 5 25 1.0 ssd_dbs25 $LCTX $RTRUNC $DDEPTH $CTRLR $TOPP ssd_ctrsa_gen.jsonl
