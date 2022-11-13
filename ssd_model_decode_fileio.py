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
from flufl.lock import Lock
from accelerate import InitProcessGroupKwargs
import datetime


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


def apply_controlling_drift(args, perturbed_inputs_diralpha):
    if args.decode_ctr_lr <= 0:
        args.ctr_loss = -1
        return perturbed_inputs_diralpha

    if args.ctr_model is None:
        ctr_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        args.ctr_model = AutoModelForSequenceClassification.from_pretrained(ctr_model_name).to(args.accelerator.device)
    optimizing_label_index = args.ctr_opt_label_idx

    for ctr_i in range(1):
        with torch.enable_grad():
            perturbed_inputs_diralpha_4ctr = perturbed_inputs_diralpha.clone()
            perturbed_inputs_diralpha_4ctr.requires_grad_()
            perturbed_inputs_simplex_4ctr = torch.nn.functional.softmax(perturbed_inputs_diralpha_4ctr, dim=-1)
            perturbed_inputs_embeds_4ctr = torch.nn.functional.linear(perturbed_inputs_simplex_4ctr, args.ctr_model.get_input_embeddings().weight.t())
            ctr_loss = -torch.nn.functional.log_softmax(args.ctr_model(inputs_embeds=perturbed_inputs_embeds_4ctr).logits, dim=-1)[:,optimizing_label_index].mean()
            args.ctr_loss = ctr_loss
            ctr_delta = -torch.autograd.grad(ctr_loss, perturbed_inputs_diralpha_4ctr)[0] # indexing 0 because the return is a tuple

        perturbed_inputs_diralpha = perturbed_inputs_diralpha + args.decode_ctr_lr * ctr_delta 
    
    return perturbed_inputs_diralpha


def logits_projection(logits, top_p, one_hot_value):
    assert len(logits.size()) == 3
    very_low_value = -10000

    # get top-p indices
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    valid_indices = nucleus.scatter(2, indices, nucleus)

    logits = logits.masked_fill(valid_indices == 0, very_low_value - one_hot_value)
    return torch.clamp(logits, max=very_low_value + one_hot_value) - very_low_value


def logits_uneven_projection(logits, top_p, one_hot_value):
    assert len(logits.size()) == 3
    very_low_value = -10000

    # get top-p indices
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    valid_indices = nucleus.scatter(2, indices, nucleus)

    filtered_logits = logits.masked_fill(valid_indices == 0, very_low_value)
    max_logits = torch.max(filtered_logits, -1, keepdim=True)[0]
    filtered_logits = filtered_logits - max_logits + one_hot_value # max logit gets +5, others keep the same diff with max logit
    return torch.clamp(filtered_logits, min=-one_hot_value)


def logits_sampling_projection(logits, top_p, one_hot_value):
    assert len(logits.size()) == 3
    very_low_value = -10000

    # get top-p indices
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    valid_indices = nucleus.scatter(2, indices, nucleus)

    filtered_logits = logits.masked_fill(valid_indices == 0, -float('Inf'))
    m = torch.distributions.categorical.Categorical(logits=filtered_logits)
    selected = m.sample()
    return 2 * one_hot_value * torch.nn.functional.one_hot(selected, logits.size(2)) - one_hot_value


def decode(args, batch_input_ids, dec_depth, total_t, model_embedding_lut, embedding_sum_layer, timestep_layer, model, tokenizer):
    batch_size = args.per_device_eval_batch_size
    if args.decode_truncate_len > 0:
        diffusion_input_ids = batch_input_ids[:, args.context_size:-args.decode_truncate_len]
    else:
        diffusion_input_ids = batch_input_ids[:, args.context_size:]
    
    # for each decode step
    assert (args.max_seq_length - args.context_size - args.decode_truncate_len) % dec_depth == 0
    unit_seq_len = int((args.max_seq_length - args.context_size - args.decode_truncate_len) / dec_depth)
    if args.context_size > 0:
        unit_context_input_ids = batch_input_ids[:, :args.context_size].clone()
    else:
        unit_context_input_ids = None
    history_decode_ids = None

    for i in range(dec_depth):
        unit_noise = args.noise_manual_scale * args.one_hot_value * torch.normal(0, 1, size=(batch_size, unit_seq_len, args.vocab_size)).to(args.accelerator.device)
        xt = unit_noise

        if unit_context_input_ids is not None:
            context_inputs_embeds = model_embedding_lut(unit_context_input_ids)
        else:
            context_inputs_embeds = None

        t_range = list(range(1, args.sigma_num_steps+1))
        t_range.reverse()
        progress_bar = tqdm(range(len(t_range)), disable=not args.accelerator.is_local_main_process)
        
        for t in t_range:
            selected_t = torch.FloatTensor([t]).repeat(batch_size).to(args.accelerator.device)
            alpha_t_bar, alpha_t_minus_bar, beta_t, beta_t_til, alpha_t = get_time_variables(selected_t, total_t, args.accelerator.device)
            beta_t_til = beta_t_til.view(batch_size, 1, 1)
            zt = args.noise_manual_scale * args.one_hot_value * torch.normal(0, 1, size=(batch_size, unit_seq_len, args.vocab_size)).to(args.accelerator.device)
            
            perturbed_inputs_diralpha = xt
            
            mean_or_protect_for_nan = True # (HACK: for the nan issue)
            if mean_or_protect_for_nan:
                perturbed_inputs_simplex = torch.nn.functional.softmax(perturbed_inputs_diralpha, dim=-1) # HACK: only for mean of dirichlet, not for sample
            else:
                perturbed_inputs_diralpha = torch.exp(perturbed_inputs_diralpha)
                dir_model = torch.distributions.dirichlet.Dirichlet(perturbed_inputs_diralpha)
                perturbed_inputs_simplex = dir_model.sample()

            # pass to the model, conditioned on the timestep as well
            perturbed_inputs_embeds = embedding_sum_layer(perturbed_inputs_simplex)
            t_progress = selected_t / total_t
            timestep_embeds = timestep_layer(t_progress.view(batch_size,1,1).repeat(1,unit_seq_len,1))

            diffusion_embeds = perturbed_inputs_embeds + timestep_embeds
            if context_inputs_embeds is not None:
                diffusion_embeds = torch.cat((context_inputs_embeds, diffusion_embeds), dim=1)
            outputs = model(inputs_embeds=diffusion_embeds, output_hidden_states=False)
            equivalent_score = outputs.logits
            if unit_context_input_ids is not None:
                equivalent_score = equivalent_score[:, unit_context_input_ids.size(1):].contiguous()

            equivalent_score = apply_controlling_drift(args, equivalent_score)
            
            if t > 1:
                sigma_t = torch.sqrt(beta_t_til)
            else:
                sigma_t = 0
            if args.loss_mode == "l2_on_z":
                raise NotImplementedError("l2_on_z samping is not implemented yet")
            else:
                if args.projection_alg == "even":
                    projected_logits = logits_projection(equivalent_score, top_p=args.projection_top_p, one_hot_value=args.one_hot_value)
                elif args.projection_alg == "sampling":
                    projected_logits = logits_sampling_projection(equivalent_score, top_p=args.projection_top_p, one_hot_value=args.one_hot_value)
                else:
                    raise ValueError("Unknown projection algorithm")

                xt = torch.sqrt(alpha_t_minus_bar).view(-1, 1, 1) * projected_logits
                xt = xt + torch.sqrt(1 - alpha_t_minus_bar).view(-1, 1, 1) * zt
            progress_bar.update(1)

            if t % 200 == 0 or t == 1:
                simplex = torch.nn.functional.softmax(xt, dim=-1) # HACK: only for mean of dirichlet, not for sample
                logger.info(f"sigma_t={sigma_t}, training_coef_at_t={torch.sqrt(1 - alpha_t_bar)}")
                logger.info(f"predicted simplex's entropy={torch.distributions.categorical.Categorical(logits=equivalent_score).entropy()}, logit_max,min,mean={torch.max(equivalent_score)},{torch.min(equivalent_score)},{torch.mean(equivalent_score)}")

                if unit_context_input_ids is not None:
                    context_sequences = tokenizer.batch_decode(unit_context_input_ids.detach().to('cpu'))
                    logger.info(f"context: {context_sequences}")
                
                real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)
                sampled_sequences = tokenizer.batch_decode(real_token_ids_list.clone().detach().to('cpu'))
                logger.info(f"t={t}: {colored(str(sampled_sequences), 'red')}")

                simplex = equivalent_score
                real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)
                sampled_sequences = tokenizer.batch_decode(real_token_ids_list.clone().detach().to('cpu'))
                logger.info(f"t={t} (before +z): {colored(str(sampled_sequences), 'green')}")

                alt_i = 1 # look at the second best candidate
                alt_real_token_ids_list = torch.topk(simplex, alt_i+1, dim=-1).indices[:, :, alt_i].view(batch_size, unit_seq_len)
                alt_sampled_sequences = tokenizer.batch_decode(alt_real_token_ids_list.clone().detach().to('cpu'))
                logger.info(f"t={t} (alt{alt_i+1}): {colored(str(alt_sampled_sequences), 'blue')}")

                logger.info(f"ctr loss: {args.ctr_loss}")
                logger.info(f"non-zero vocab: {torch.count_nonzero(projected_logits > -args.one_hot_value+0.0001) / simplex.size(0) / simplex.size(1)} out of {torch.numel(projected_logits) / simplex.size(0) / simplex.size(1)}")
        
        unit_context_input_ids = torch.cat((unit_context_input_ids, real_token_ids_list), dim=1)
        if history_decode_ids is None:
            history_decode_ids = real_token_ids_list
        else:
            history_decode_ids = torch.cat((history_decode_ids, real_token_ids_list), dim=1)

    if args.context_size > 0:
        init_context_input_ids = batch_input_ids[:, :args.context_size].clone()
        context_sequences = tokenizer.batch_decode(init_context_input_ids.detach().to('cpu'))
    else:
        init_context_input_ids = None
        context_sequences = None
    gold_sequences = tokenizer.batch_decode(diffusion_input_ids.clone().detach().to('cpu'))
    sampled_sequences = tokenizer.batch_decode(history_decode_ids.clone().detach().to('cpu'))
    logger.info(f"context: {context_sequences}")
    logger.info(f"gold: {colored(str(gold_sequences), 'yellow')}")
    logger.info(f"t={t}: {colored(str(sampled_sequences), 'red')}")

    return history_decode_ids, init_context_input_ids, diffusion_input_ids, sampled_sequences, context_sequences, gold_sequences


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
    # for computing influence scores w.r.t. the querying file
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
    )
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
    parser.add_argument(
        "--out_fn", type=str, default="_sample_gen.jsonl", help="",
    )
    parser.add_argument(
        "--projection_top_p", type=float, default=0.2, help="",
    )
    parser.add_argument(
        "--projection_alg", type=str, default="even", help="",
    ) # even, sampling
    parser.add_argument(
        "--ctr_opt_label_idx", type=int, default=0, help="",
    ) # 0 (neg in sentiment), 2 (pos in sentiment)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=259200))])
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

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

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

    # # If we want to use a non-existing architecture, we can do it here
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
    if args.model_name_or_path == args.output_dir:
        _stdict = torch.load(os.path.join(args.output_dir, "embed_sum_layer.pt"))
        _stdict = dict((_k[7:], _stdict[_k]) if _k.startswith("module.") else (_k, _stdict[_k]) for _k in _stdict)
        # _stdict = dict((f"module.{_k}", _stdict[_k]) for _k in _stdict)
        embedding_sum_layer.load_state_dict(_stdict)
        _stdict = torch.load(os.path.join(args.output_dir, "timestep_layer.pt"))
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
    if args.train_mode == "resume": # resuming job would still break the strict reproducibility, since we are not saving noise states
        accelerator.load_state(os.path.join(args.output_dir, 'accelerate_ckpt'))
        with open(os.path.join(args.output_dir, "completed_steps.txt"), 'r') as f:
            completed_steps = int(f.read())
    elif args.train_mode == "train":
        if os.path.exists(os.path.join(args.output_dir, 'accelerate_ckpt')):
            logger.info("training probably interrupted, should change mode to resume for the next run")
            return 0
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
        raise ValueError("Training or resuming is disabled here")

    ##########################################

    out_json_fn = os.path.join(args.output_dir, f"ctx{args.decode_context_size}_trunc{args.decode_truncate_len}_depth{args.decode_depth}_ctrlr{args.decode_ctr_lr}_step{args.sigma_num_steps}_topp{args.projection_top_p}_decalg{args.projection_alg}_ctridx{args.ctr_opt_label_idx}_" + args.out_fn)

    # Decoding, includes hardcode for now
    if args.train_mode == "decode":
        _stdict = torch.load(os.path.join(args.output_dir, "embed_sum_layer.pt"))
        # # Use this when running without accelerator
        # _stdict = dict((_k[7:], _stdict[_k]) if _k.startswith("module.") else (_k, _stdict[_k]) for _k in _stdict)
        # Use this when running with accelerator
        _stdict = dict((f"module.{_k}", _stdict[_k]) if not _k.startswith("module.") else (_k, _stdict[_k]) for _k in _stdict)
        embedding_sum_layer.load_state_dict(_stdict)

        _stdict = torch.load(os.path.join(args.output_dir, "timestep_layer.pt"))
        # # Use this when running without accelerator
        # _stdict = dict((_k[7:], _stdict[_k]) if _k.startswith("module.") else (_k, _stdict[_k]) for _k in _stdict)
        # Use this when running with accelerator
        _stdict = dict((f"module.{_k}", _stdict[_k]) if not _k.startswith("module.") else (_k, _stdict[_k]) for _k in _stdict)
        timestep_layer.load_state_dict(_stdict)

        model.eval()

        args.sigma_noise_scale = 1.0
        args.interpolation_with_prev = 0.0

        args.context_size = args.decode_context_size
        args.one_hot_value = one_hot_value
        args.vocab_size = vocab_size
        args.accelerator = accelerator
        args.ctr_model = None

        if "interactive" in args.remove_noise_mode:
            args.orig_decode_truncate_len = args.decode_truncate_len
            with torch.no_grad():
                while True:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        prompt = input("Type in your prompt: ")
                        if prompt:
                            if prompt[0] != " ": # add a space if the user didn't
                                prompt = f" {prompt}"
                            input_ids = torch.LongTensor(tokenizer.encode(prompt, add_special_tokens=False)).to(args.accelerator.device)
                            args.context_size = len(input_ids)
                            args.decode_truncate_len = args.orig_decode_truncate_len - args.context_size # compensates for the unknown input context size
                            input_ids = input_ids.unsqueeze(0)
                            history_decode_ids, context_input_ids, diffusion_input_ids, sampled_sequences, context_sequences, gold_sequences = \
                                decode(args, input_ids, args.decode_depth, total_t, model_embedding_lut, embedding_sum_layer, timestep_layer, model, tokenizer)
                        else:
                            breakpoint()
                    accelerator.wait_for_everyone()
        
        elif "fin" in args.remove_noise_mode:
            assert args.remove_noise_mode[0] == "fin"
            assert os.path.exists(args.remove_noise_mode[1])
            fin_path = args.remove_noise_mode[1]
            fin_data = []
            with open(fin_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line:
                        fin_data.append(json.loads(line))

            export_list = []
            args.orig_decode_truncate_len = args.decode_truncate_len
            with torch.no_grad():
                for step, _fd in enumerate(fin_data): # only support batch size 1 since the context size can be different across lines
                    accelerator.wait_for_everyone()
                    ctx_field_name = 'context_string'
                    assert ctx_field_name in _fd
                    assert args.per_device_eval_batch_size == 1

                    input_ids = torch.LongTensor(tokenizer.encode(_fd[ctx_field_name], add_special_tokens=False)).to(args.accelerator.device)
                    args.context_size = len(input_ids)
                    args.decode_truncate_len = args.orig_decode_truncate_len - args.context_size # Han: this compensates for the unknown input context size
                    input_ids = input_ids.unsqueeze(0)

                    repeat_sample = 20 # Han: currently change here manually
                    for _r in range(repeat_sample):
                        history_decode_ids, context_input_ids, diffusion_input_ids, sampled_sequences, context_sequences, gold_sequences = \
                            decode(args, input_ids, args.decode_depth, total_t, model_embedding_lut, embedding_sum_layer, timestep_layer, model, tokenizer)
                        if _r == 0: # first sample
                            # export to jsonl
                            for _i in range(args.per_device_eval_batch_size):
                                export_dict = dict()
                                export_dict['context_len'] = args.context_size
                                export_dict['context'] = context_input_ids.tolist()[_i]
                                export_dict['context_string'] = context_sequences[_i]
                                export_dict['len'] = args.max_seq_length - args.context_size - args.decode_truncate_len
                                export_dict['tokens'] = [history_decode_ids.tolist()[_i]]
                                export_dict['string'] = [sampled_sequences[_i]]
                                export_dict['gold_tokens'] = diffusion_input_ids.tolist()[_i]
                                export_dict['gold_string'] = gold_sequences[_i]
                                export_list.append(export_dict)
                        else:
                            for _i in range(args.per_device_eval_batch_size):
                                export_list[-(args.per_device_eval_batch_size - _i)]['tokens'].append(history_decode_ids.tolist()[_i])
                                export_list[-(args.per_device_eval_batch_size - _i)]['string'].append(sampled_sequences[_i])

            if accelerator.is_main_process:
                if os.path.exists(out_json_fn):
                    os.remove(out_json_fn)
                    logger.info(f"Cleaning existing {out_json_fn}")
            accelerator.wait_for_everyone()
            with Lock(out_json_fn + '.lock', lifetime=120):
                with open(out_json_fn, mode="a") as f_out:
                    for export in export_list:
                        f_out.write(json.dumps(export))
                        f_out.write("\n")
            accelerator.wait_for_everyone()

        elif "dump_dataloader" in args.remove_noise_mode: # some tokens like </s> might be wrongly parsed; for formal evaluation, we directly load the dataloader, not load from jsonl
            # breakpoint()
            dump_json_fn = os.path.join(args.output_dir, f"owt_eval_input_ctx{args.decode_context_size}.jsonl")
            export_list = []
            with torch.no_grad():
                for step, batch in enumerate(eval_dataloader):
                    accelerator.wait_for_everyone()

                    if args.decode_truncate_len > 0:
                        ctx_input_ids = batch['input_ids'][:, :args.context_size]
                    else:
                        ctx_input_ids = batch['input_ids'][:, :args.context_size]
                    context_sequences = tokenizer.batch_decode(ctx_input_ids.detach().to('cpu'))

                    for _i in range(args.per_device_eval_batch_size):
                        export_dict = dict()
                        export_dict["context_string"] = context_sequences[_i]
                        export_list.append(export_dict)

                    num_exports = 5 # Han: currently change here manually, multiply this with effective eval bs
                    if (step + 1) == num_exports:
                        if accelerator.is_main_process:
                            if os.path.exists(dump_json_fn):
                                os.remove(dump_json_fn)
                                logger.info(f"Cleaning existing {dump_json_fn}")
                        accelerator.wait_for_everyone()
                        with Lock(dump_json_fn + '.lock', lifetime=120):
                            with open(dump_json_fn, mode="a") as f_out:
                                for export in export_list:
                                    f_out.write(json.dumps(export))
                                    f_out.write("\n")
                        accelerator.wait_for_everyone()
                        break
        
        else:
            raise ValueError("correct mode should be included in remove_noise_mode, a string separated by pipes")


if __name__ == "__main__":
    main()
