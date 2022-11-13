import pandas as pd
from pathlib import Path
import os
import numpy as np

import click
import math
import torch
import torch.nn as nn

import argparse
import json
import os
import operator
import logging
import random
import time

from functools import partial
from collections import Counter
from scipy import stats
from multiprocessing.pool import Pool
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig, TextClassificationPipeline

logger = logging.getLogger(__name__)


@click.command()
@click.option('--model_name', required=True, type=str, help='huggingface model name to decode from')
@click.option('--input_file', required=True, type=str, help='a jsonl file with prompts')
@click.option('--output_file', required=True, type=str, help='filename to write the results to')
@click.option('--batch_size', required=False, type=int, default=25, help='huggingface model name to decode from')
@click.option('--repeat_sample', required=False, type=int, default=5, help='huggingface model name to decode from')
@click.option('--max_length', required=False, type=int, default=100, help='huggingface model name to decode from')
@click.option('--top_p', required=False, type=float, default=1.0, help='huggingface model name to decode from')
@click.option('--typical_p', required=False, type=float, default=1.0, help='huggingface model name to decode from')
@click.option('--seed', required=False, type=int, default=2022, help='huggingface model name to decode from')
def main(model_name, input_file, output_file, batch_size, repeat_sample, max_length, top_p, typical_p, seed):
    assert batch_size % repeat_sample == 0, 'batch_size must be divisible by repeat_sample'
    assert os.path.exists(input_file)
    output_dir = Path(os.path.dirname(input_file))
    generations_df = pd.read_json(input_file, lines=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### calculate quality metrics
    roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = roberta_tokenizer.pad_token # to avoid an error
    tokenizer.bos_token = roberta_tokenizer.bos_token
    tokenizer.eos_token = roberta_tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    print("model and tokenizer loaded")
    model = model.eval()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Fluency
    fo = open(output_dir / output_file, 'w') #creating the file

    print("read the file")
    st=time.time()
    text_batch = []
    text_outputs = []
    with torch.no_grad():
        for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Generating outputs'):
            for _j in range(repeat_sample):
                text_batch.append(row.context_string)
            if len(text_batch) == batch_size or (i+1) == len(generations_df.index):
                # print(i+1)
                torch.cuda.empty_cache()
                tokens = tokenizer(text_batch, return_tensors="pt", padding=True)
                
                outputs = model.generate(input_ids=tokens['input_ids'].to(device), attention_mask=tokens['attention_mask'].to(device),
                                pad_token_id=tokenizer.eos_token_id,
                                max_length=max_length, min_length=max_length, do_sample=True, top_p=top_p, typical_p=typical_p
                                )
                                
                # Han: save (0) baseline full output, (1) original context ids, (2) baseline newly generated ids, (3) baseline newly generated text
                text_outputs += [(tokenizer.decode(t, skip_special_tokens=True), \
                    tokens['input_ids'][_i].tolist(), \
                    outputs[_i].tolist()[len(tokens['input_ids'][_i]):], \
                    tokenizer.decode(outputs[_i][len(tokens['input_ids'][_i]):], skip_special_tokens=True)) for _i, t in enumerate(outputs)]
                
                del tokens
                del outputs
                text_batch = []                
            
            if (i + 1) % 100 == 0:
                print(f"done {i+1} contexts, took {(time.time()-st)/(i+1)} seconds per context, total generations {len(text_outputs)}", flush=True)

        assert len(text_outputs) == len(generations_df.index) * repeat_sample
        # print(text_outputs)
        # print(len(generations_df.index))
        generations_df[f'{model_name}_outputs'] = ""
        for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Writing outputs'):
            # generations_df.loc[i, f'{model_name}_outputs'] = [{'text': text_outputs[j][0]} for j in range(i*repeat_sample, (i+1)*repeat_sample)] # don't really need this
            
            # Han: overwrite original fields for easy evaluation
            generations_df[f'context'][i] = text_outputs[i*repeat_sample][1]
            generations_df[f'tokens'][i] = [text_outputs[j][2] for j in range(i*repeat_sample, (i+1)*repeat_sample)] # note this may have mismatched token ids compared to our roberta-based model
            generations_df[f'string'][i] = [text_outputs[j][3] for j in range(i*repeat_sample, (i+1)*repeat_sample)]
            # print(row)
        
        fo.write(generations_df.to_json(orient='records', lines=True))
    

if __name__ == '__main__':
    main()