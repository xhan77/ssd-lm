#!/usr/bin/bash
trap "kill 0" EXIT

script_role="host"
global_seed=$1 # inline param, 2021, 2022, etc
single_device_cuda="0" # inline param, "0", "1", etc
multi_device_cuda=$2 # inline param, "0,1,2,3", "0", etc
hf_cache="/private/home/xhan77/.cache/huggingface"
core_lm_name="roberta-large"
main_log_dir="/private/home/xhan77/ssd-lm/logging"

# load from created dataset
interpret_dataset_tokenized_path="${main_log_dir}/openwebtext_processed_pct100_blk200"

# data hyperparameters
global_max_seq_len=200
####

# retrain
retrain_num_train_epochs=10000 # just a placeholder, use max train steps
retrain_per_device_train_batch_size=24
retrain_per_device_eval_batch_size=1
retrain_learning_rate=$3
retrain_weight_decay=0.01
retrain_gradient_accumulation_steps=8
retrain_num_warmup_steps=2000
retrain_max_train_steps=100000

sigma_num_steps=$4
loss_mode=$5
remove_noise_mode=$6
pa=$7
cs=0 # placeholder
dbs=$8
precision=$9 # no or fp16
noise_manual_scale=${10}
train_mode=${11}

################ START ################

# available_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
available_port=29510
main_node_name=$(scontrol show hostnames $SLURM_JOB_NODELIST | sort | head -n 1)
main_ip_address=$(python -c 'import sys; import socket; ip=socket.gethostbyname(sys.argv[1]); print(ip)' ${main_node_name})

CUDA_VISIBLE_DEVICES=${multi_device_cuda} HF_HOME=${hf_cache} accelerate launch \
    --multi_gpu --mixed_precision ${precision} \
    --num_processes $((8*SLURM_JOB_NUM_NODES)) --num_machines ${SLURM_JOB_NUM_NODES} --machine_rank ${SLURM_NODEID} \
    --main_process_ip ${main_ip_address} --main_process_port ${available_port} \
    --num_cpu_threads_per_process 2 \
    ssd_model_train.py \
    --max_seq_length ${global_max_seq_len} \
    --model_name_or_path ${core_lm_name} \
    --num_train_epochs ${retrain_num_train_epochs} \
    --per_device_train_batch_size ${retrain_per_device_train_batch_size} \
    --per_device_eval_batch_size ${retrain_per_device_eval_batch_size} \
    --learning_rate ${retrain_learning_rate} \
    --weight_decay ${retrain_weight_decay} \
    --gradient_accumulation_steps ${retrain_gradient_accumulation_steps} \
    --num_warmup_steps ${retrain_num_warmup_steps} \
    --max_train_steps ${retrain_max_train_steps} \
    --seed ${global_seed} \
    --use_slow_tokenizer \
    --output_dir ${main_log_dir}/ssd_dbs${dbs} \
    --loss_mode ${loss_mode} \
    --remove_noise_mode ${remove_noise_mode} \
    --hardcoded_pseudo_diralpha ${pa} \
    --context_size ${cs} \
    --decoding_block_size ${dbs} \
    --sigma_num_steps ${sigma_num_steps} \
    --noise_manual_scale ${noise_manual_scale} \
    --tokenized_data_file_path ${interpret_dataset_tokenized_path} \
    --if_create_tokenized_data_file "no" \
    --train_mode ${train_mode}