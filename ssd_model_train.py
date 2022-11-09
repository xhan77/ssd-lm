#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

import argparse
from cmath import exp
import logging
import math
from multiprocessing.sharedctypes import Value
import os
import random
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm.auto import tqdm

import transformers
import accelerate
from accelerate import Accelerator, DistributedType
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

import numpy as np
import pickle
from filelock import FileLock
import dill
from datasets import load_from_disk
from datasets import concatenate_datasets
from termcolor import colored
import time
import json


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def get_time_variables(t, total_t, device): # according to https://arxiv.org/pdf/2102.09672.pdf

    def ft(small_t, big_t, s=1e-4):
        return torch.cos((small_t / big_t + s) / (1 + s) * math.pi / 2) ** 2

    alpha_t_bar = ft(t, total_t) / ft(torch.zeros(t.shape).to(device), total_t)
    alpha_t_minus_bar = ft(t-1, total_t) / ft(torch.zeros(t.shape).to(device), total_t)
    beta_t = 1 - (alpha_t_bar / alpha_t_minus_bar)
    beta_t_til = (1 - alpha_t_minus_bar) / (1 - alpha_t_bar) * beta_t
    alpha_t = 1 - beta_t
    return alpha_t_bar, alpha_t_minus_bar, beta_t, beta_t_til, alpha_t


def analyze_perturbed_text_properties(args, selected_t, alpha_t_bar, target_token_ids_list, perturbed_inputs_simplex, vocab2freq):
    bs = perturbed_inputs_simplex.size(0)
    sl = perturbed_inputs_simplex.size(1)
    vs = perturbed_inputs_simplex.size(2)
    real_token_ids_list = torch.argmax(perturbed_inputs_simplex, dim=-1).view(bs, sl)
    assert bs == 1 # Han: a bit slow but anyway it's just an analysis

    subtitution_rate = target_token_ids_list.view(-1).ne(real_token_ids_list.view(-1)).sum().item() * 1.0 / (bs * sl)
    simplex_entropy = torch.distributions.categorical.Categorical(perturbed_inputs_simplex).entropy().mean().item()
    simplex_nll = torch.nn.functional.nll_loss(torch.log(perturbed_inputs_simplex.view(-1, vs)), target_token_ids_list.view(-1)).item()
    orig_token_freq = np.mean([vocab2freq[tid.item()] for tid in target_token_ids_list.view(-1)])
    noisy_token_freq = np.mean([vocab2freq[tid.item()] for tid in real_token_ids_list.view(-1)])

    args.noise_analysis_list.append({'timestep': selected_t.item(),
                                    'sqrt_one_minus_alpha_t_bar': torch.sqrt(1 - alpha_t_bar).item(),
                                    'subtitution_rate': subtitution_rate,
                                    'simplex_entropy': simplex_entropy,
                                    'simplex_nll': simplex_nll,
                                    'orig_token_freq': orig_token_freq,
                                    'noisy_token_freq': noisy_token_freq})


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    # Han: many arguments below will not be used, but keeping for future edits
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library). For example, Wikipedia.",
    )
    parser.add_argument(
        "--additional_dataset_name",
        type=str,
        default=None,
        help="The name of the additional dataset to use (via the datasets library). For example, BookCorpus.",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--raw_data_percentage",
        default=100,
        help="The percentage of raw data used as the train set",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--data_cap",
        type=int,
        default=2,
        help="Max number of data for which we will save graidents.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.0, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--no_save_grads", action="store_true", help="Whether to save gradients to a file.")
    # for computing influence scores w.r.t. the querying file (not relevant for SSD-LM)
    parser.add_argument(
        "--query_file", type=str, default=None, help="A pickle file containing gradient information from the querying data."
    )
    parser.add_argument(
        "--query_data_cap", type=int, default=None, help="Max number of data for which we will save gradients.",
    )
    parser.add_argument("--influence_metric", type=str, default=None, help="Metric for computing the gradients.")
    parser.add_argument("--init_blank_language_model", action="store_true", help="Whether or not to use a completely blank LM.")
    parser.add_argument(
        "--tokenized_data_file_path", type=str, default=None, help="Path of the tokenized data file."
    )
    parser.add_argument(
        "--if_create_tokenized_data_file", type=str, default=None, help="Whether to create a new tokenized data file (yes or no)."
    )
    parser.add_argument(
        "--sigma_start_value", type=float, default=-1, help="",
    )
    parser.add_argument(
        "--sigma_end_value", type=float, default=-1, help="",
    )
    parser.add_argument(
        "--sigma_num_steps", type=int, default=1000, help="",
    )
    parser.add_argument(
        "--loss_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--remove_noise_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--hardcoded_pseudo_diralpha", type=float, default=3, help="",
    ) # this is the one-hot value (simplex from logits can be seen as the mean of a Dirichlet distribution)
    parser.add_argument(
        "--context_size", type=int, default=0, help="",
    )
    parser.add_argument(
        "--decoding_block_size", type=int, default=25, help="",
    )
    parser.add_argument(
        "--train_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--noise_manual_scale", type=float, default=1, help="",
    )
    parser.add_argument(
        "--decode_context_size", type=int, default=25, help="",
    ) # how many to cut from left
    parser.add_argument(
        "--decode_truncate_len", type=int, default=50, help="",
    ) # how many to cut from right
    parser.add_argument(
        "--decode_depth", type=int, default=2, help="",
    )
    parser.add_argument(
        "--decode_ctr_lr", type=float, default=0.0, help="",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        # set_seed(args.seed)
        accelerate.utils.set_seed(args.seed, device_specific=True) # differ slightly for each device

    # HACK: we can pass in "resume" mode, but if output_dir doesn't exist, we change to "train" mode
    if args.train_mode == "resume" and not os.path.exists(args.output_dir):
        args.train_mode = "train"
    time.sleep(5)
    accelerator.wait_for_everyone()
    time.sleep(5)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # under decode mode, will load model from the output_dir
    if args.train_mode == "decode":
        args.model_name_or_path = args.output_dir
        logger.info(f"Overwriting model_name_or_path ({args.model_name_or_path}) with {args.output_dir}")
    elif args.train_mode == "train":
        train_losses_log_file = os.path.join(args.output_dir, "training_losses.txt")
        if accelerator.is_main_process:
            if os.path.exists(train_losses_log_file):
                os.remove(train_losses_log_file)
                logger.info(f"Cleaning existing {train_losses_log_file}")
        accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    assert args.use_slow_tokenizer == True
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    jump = False
    if args.tokenized_data_file_path and args.if_create_tokenized_data_file:
        if args.if_create_tokenized_data_file == "no":
            tokenized_datasets = load_from_disk(args.tokenized_data_file_path)
            jump = True
        elif args.if_create_tokenized_data_file == "yes":
            raise ValueError("should not create dataset in this train file")
            if accelerator.is_main_process:
                pass
        else:
            raise ValueError("check args.if_create_tokenized_data_file")

    full_dataset = tokenized_datasets["train"]
    validation_ratio = 0.01 # 1 pct of the data is used for validation
    validation_len = int(len(full_dataset) * validation_ratio)
    train_len = len(full_dataset) - validation_len
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_len, validation_len], generator=torch.Generator().manual_seed(42)) # fixing seed here

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator
    # This one will take care of randomly masking the tokens.
    assert args.mlm_probability == 0 # diffusion model does not use [MASK]
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size, generator=torch.Generator().manual_seed(42)
    )

    ########

    # # If we want to use a non-existing architecture, we can do it here, e.g.,
    # config.hidden_size = 1600
    # config.intermediate_size = 4096
    # config.max_position_embeddings = 128
    # config.num_attention_heads = 25
    # config.num_hidden_layers = 48

    if args.init_blank_language_model:
        model = AutoModelForMaskedLM.from_config(config)
    elif args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        raise ValueError("specify --init_blank_language_model")
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    vocab_size = model.get_input_embeddings().weight.size(0)
    hidden_size = model.get_input_embeddings().weight.size(1)
    embedding_sum_layer = torch.nn.Linear(vocab_size, hidden_size, bias=False)
    with torch.no_grad():
        embedding_sum_layer.weight.copy_(torch.transpose(model.get_input_embeddings().weight.clone(), 0, 1))
    timestep_layer = torch.nn.Linear(1, hidden_size, bias=True)

    # load in our customized modules if necessary
    if os.path.exists(args.model_name_or_path):
        _stdict = torch.load(os.path.join(args.model_name_or_path, "embed_sum_layer.pt"))
        _stdict = dict((_k[7:], _stdict[_k]) if _k.startswith("module.") else (_k, _stdict[_k]) for _k in _stdict)
        # _stdict = dict((f"module.{_k}", _stdict[_k]) for _k in _stdict)
        embedding_sum_layer.load_state_dict(_stdict)
        _stdict = torch.load(os.path.join(args.model_name_or_path, "timestep_layer.pt"))
        _stdict = dict((_k[7:], _stdict[_k]) if _k.startswith("module.") else (_k, _stdict[_k]) for _k in _stdict)
        # _stdict = dict((f"module.{_k}", _stdict[_k]) for _k in _stdict)
        timestep_layer.load_state_dict(_stdict)

    # Optimizer
    frozen = []
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if (not any(fr in n for fr in frozen)) and (not any(nd in n for nd in no_decay))],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if (not any(fr in n for fr in frozen)) and (any(nd in n for nd in no_decay))],
            "weight_decay": 0.0,
        },
        {
            "params": [p for p in embedding_sum_layer.parameters()],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for p in timestep_layer.parameters()],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    assert args.max_train_steps is not None

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, embedding_sum_layer, timestep_layer, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, embedding_sum_layer, timestep_layer, optimizer, train_dataloader, eval_dataloader
    )

    # Register the LR scheduler
    accelerator.register_for_checkpointing(lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    # if accelerator.distributed_type == DistributedType.TPU:
    #     model.tie_weights()

    # Save accelerator state
    accelerator.wait_for_everyone()
    if args.train_mode == "resume": # resuming job could still break an exact reproducibility, since we are not saving noise states
        accelerator.load_state(os.path.join(args.output_dir, 'accelerate_ckpt'))
        with open(os.path.join(args.output_dir, "completed_steps.txt"), 'r') as f:
            completed_steps = int(f.read())
    elif args.train_mode == "train":
        if os.path.exists(os.path.join(args.output_dir, 'accelerate_ckpt')):
            logger.info("training probably interrupted, should change mode to resume for the next run")
            # return 0 # just give warnings, do not interrupt
        accelerator.save_state(os.path.join(args.output_dir, 'accelerate_ckpt'))
        completed_steps = 0
    elif args.train_mode == "decode":
        pass
    else:
        raise ValueError("train_mode must be one of 'train', 'resume', 'decode'")

    model_embedding_lut = accelerator.unwrap_model(model).get_input_embeddings()

    t_list = list(range(1, args.sigma_num_steps+1))
    total_t = args.sigma_num_steps
    one_hot_value = args.hardcoded_pseudo_diralpha # for a pseudo one-hot encoding for alpha

    args.remove_noise_mode = args.remove_noise_mode.split('|')
    args.noise_analysis_list = list()

    if args.train_mode == "train" or args.train_mode == "resume":
        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f"  Completed optimization steps = {completed_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps - completed_steps), disable=not accelerator.is_local_main_process)

        batch_size = args.per_device_train_batch_size

        # begin training
        train_losses = []
        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                ctx_low = 1 # using 0 would probably cause hanging issue (parallel device waiting for syncing gradients?)
                ctx_high = args.max_seq_length - args.decoding_block_size + 1
                args.context_size = np.random.randint(low=ctx_low, high=ctx_high) # min (1 dec block size), max (max seq length - 1 dec block size)
                
                seq_len = args.decoding_block_size

                # split the batch in to the context part and diffusion part
                diffusion_input_ids = batch['input_ids'][:, args.context_size:args.context_size+seq_len]
                if args.context_size > 0:
                    context_input_ids = batch['input_ids'][:, :args.context_size]
                    context_inputs_embeds = model_embedding_lut(context_input_ids)
                else:
                    context_inputs_embeds = None

                # build alpha according to a pseudo one-hot encoding
                inputs_diralpha = 2 * one_hot_value * torch.nn.functional.one_hot(diffusion_input_ids, vocab_size) - one_hot_value

                ######## NOISE ADDITION ########
                selected_t = torch.FloatTensor(np.random.choice(t_list, batch_size, replace=True)).to(accelerator.device)
                alpha_t_bar, alpha_t_minus_bar, beta_t, beta_t_til, alpha_t = get_time_variables(selected_t, total_t, accelerator.device)
                alpha_t_bar = alpha_t_bar.view(batch_size, 1, 1)

                unit_noise = args.noise_manual_scale * one_hot_value * torch.normal(0, 1, size=inputs_diralpha.shape).to(accelerator.device)
                if 'no_z' in args.remove_noise_mode:
                    raise ValueError("no_z is disabled for now")
                    unit_noise = unit_noise * 0

                if 'biased_z' in args.remove_noise_mode:
                    raise ValueError("biased_z is disabled for now")
                else:
                    perturbed_inputs_diralpha_noexp = torch.sqrt(alpha_t_bar) * inputs_diralpha + torch.sqrt(1 - alpha_t_bar) * unit_noise

                # sample the input simplex from dirichlet distribution
                if 'no_dir' in args.remove_noise_mode:
                    inputs_diralpha = torch.exp(inputs_diralpha) # dirichlet's alpha cannot be negative
                    mean_or_protect_for_nan = True # HACK: for the overflow issue
                    if mean_or_protect_for_nan:
                        perturbed_inputs_simplex = torch.nn.functional.softmax(perturbed_inputs_diralpha_noexp, dim=-1) # HACK: only for mean of dirichlet, not for sample
                    else:
                        perturbed_inputs_diralpha = torch.exp(perturbed_inputs_diralpha_noexp) # dirichlet's alpha cannot be negative, TODO: but leads to overflow issue sometimes
                        dir_model = torch.distributions.dirichlet.Dirichlet(perturbed_inputs_diralpha)
                        perturbed_inputs_simplex = dir_model.mean # Han: choose between .sample() and .mean?
                else:
                    raise ValueError("have to specify no_dir now")
                    inputs_diralpha = torch.exp(inputs_diralpha) # dirichlet's alpha cannot be negative
                    perturbed_inputs_diralpha = torch.exp(perturbed_inputs_diralpha_noexp) # dirichlet's alpha cannot be negative
                    dir_model = torch.distributions.dirichlet.Dirichlet(perturbed_inputs_diralpha)
                    perturbed_inputs_simplex = dir_model.sample() # Han: choose between .sample() and .mean?

                if 'debug' in args.remove_noise_mode:
                    raise ValueError("debug is disabled for now")
                
                # pass to the model, conditioned on the timestep as well
                perturbed_inputs_embeds = embedding_sum_layer(perturbed_inputs_simplex)
                t_progress = selected_t / total_t
                timestep_embeds = timestep_layer(t_progress.view(batch_size,1,1).repeat(1,seq_len,1))

                diffusion_embeds = perturbed_inputs_embeds + timestep_embeds
                if context_inputs_embeds is not None:
                    diffusion_embeds = torch.cat((context_inputs_embeds, diffusion_embeds), dim=1)
                outputs = model(inputs_embeds=diffusion_embeds, output_hidden_states=False)
                equivalent_score = outputs.logits
                equivalent_score = equivalent_score[:, args.context_size:].contiguous()

                # what we want to do with the output, loss mode kl or bhatt, the lower the better, reference: https://github.com/cran/Compositional/blob/master/R/kl.diri.R
                if args.loss_mode == "kl":
                    raise ValueError("kl is disabled for now")
                    a = inputs_diralpha.view(-1, vocab_size)
                    b = torch.exp(equivalent_score).view(-1, vocab_size)
                    a0 = a.sum(dim=-1)
                    b0 = b.sum(dim=-1)
                    loss = torch.sum( (a - b) * ( torch.digamma(a) - torch.digamma(a0).view(batch_size * seq_len, 1) ) , dim=-1) + torch.sum( torch.lgamma(b) - torch.lgamma(a) , dim=-1) + torch.lgamma(a0) - torch.lgamma(b0)
                    loss = torch.mean(loss)
                elif args.loss_mode == "bhatt":
                    raise ValueError("bhatt is disabled for now")
                    a = inputs_diralpha.view(-1, vocab_size)
                    b = torch.exp(equivalent_score).view(-1, vocab_size)
                    a0 = a.sum(dim=-1)
                    b0 = b.sum(dim=-1)
                    loss = torch.lgamma( 0.5 * torch.sum(a + b, dim=-1) ) + 0.5 * torch.sum( torch.lgamma(a) + torch.lgamma(b) , dim=-1) - torch.sum( torch.lgamma( 0.5 * (a + b) ) , dim=-1) - 0.5 * ( torch.lgamma(a0) + torch.lgamma(b0) )
                    loss = torch.mean(loss)
                elif args.loss_mode == "xe":
                    loss = torch.nn.functional.cross_entropy(equivalent_score.view(-1, vocab_size), diffusion_input_ids.contiguous().view(-1))
                    loss = torch.mean(loss)
                elif args.loss_mode == "l2_on_z":
                    raise ValueError("l2_on_z is disabled for now")
                    diff_to_be_normed = unit_noise - equivalent_score
                    loss = torch.mean(torch.linalg.norm(diff_to_be_normed.view(batch_size * seq_len * vocab_size, -1), ord=2, dim=-1) ** 2)
                else:
                    raise ValueError("check loss_mode")

                train_losses.append(loss.item())
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)

                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if (step + 1) % (args.gradient_accumulation_steps * 100) == 0 or step == len(train_dataloader) - 1:
                    mean_loss = np.mean(train_losses)
                    logger.info(f"train loss: {mean_loss}")
                    train_losses = []
                    if (step + 1) % (args.gradient_accumulation_steps * 1000) == 0 or step == len(train_dataloader) - 1:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                        if accelerator.is_main_process:
                            tokenizer.save_pretrained(args.output_dir)
                            torch.save(accelerator.unwrap_model(embedding_sum_layer).state_dict(), os.path.join(args.output_dir, "embed_sum_layer.pt"))
                            torch.save(accelerator.unwrap_model(timestep_layer).state_dict(), os.path.join(args.output_dir, "timestep_layer.pt"))
                            with open(os.path.join(args.output_dir, "completed_steps.txt"), 'w') as f:
                                f.write(f"{completed_steps}")
                            with open(os.path.join(args.output_dir, "training_losses.txt"), 'a') as f:
                                f.write(f"{completed_steps} : {mean_loss}\n")
                        logger.info(f"saved model at completed steps {completed_steps}")
                        accelerator.save_state(os.path.join(args.output_dir, 'accelerate_ckpt'))

                        # separately save some model checkpoints (not in use if interval > max_steps_for_one_epoch, fix later)
                        _log_interval = 100000 # probably too large and not in use
                        if (step + 1) % (args.gradient_accumulation_steps * _log_interval) == 0:
                            ckpt_dir = os.path.join(args.output_dir, f"ckpt{int((step+1)/(args.gradient_accumulation_steps*_log_interval))}/")
                            if accelerator.is_main_process:
                                os.makedirs(ckpt_dir, exist_ok=True)
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(ckpt_dir, save_function=accelerator.save)
                            if accelerator.is_main_process:
                                tokenizer.save_pretrained(ckpt_dir)
                                torch.save(accelerator.unwrap_model(embedding_sum_layer).state_dict(), os.path.join(ckpt_dir, "embed_sum_layer.pt"))
                                torch.save(accelerator.unwrap_model(timestep_layer).state_dict(), os.path.join(ckpt_dir, "timestep_layer.pt"))
                            logger.info(f"Separately logged the model at completed steps {completed_steps}")

                if completed_steps >= args.max_train_steps:
                    break
            if completed_steps >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            torch.save(accelerator.unwrap_model(embedding_sum_layer).state_dict(), os.path.join(args.output_dir, "embed_sum_layer.pt"))
            torch.save(accelerator.unwrap_model(timestep_layer).state_dict(), os.path.join(args.output_dir, "timestep_layer.pt"))
        logger.info(f"TRAINING FINISHED!!! Saved model at completed steps {completed_steps}")

    ##########################################

    # Decoding, now in a separate file
    if args.train_mode == "decode" and accelerator.is_main_process:
        raise ValueError("Decoding is disabled for now")


if __name__ == "__main__":
    main()
