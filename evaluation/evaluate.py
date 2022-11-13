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

from functools import partial
from collections import Counter
from scipy import stats
from multiprocessing.pool import Pool
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig, TextClassificationPipeline, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

def bleu_i(weights, all_sentences, smoothing_function, i):
    # noinspection PyTypeChecker
    return sentence_bleu(
        references=all_sentences[:i] + all_sentences[i + 1:],
        hypothesis=all_sentences[i],
        weights=weights,
        smoothing_function=smoothing_function)


def conditional_perplexity(generations_df, model, tokenizer, device='cuda', write_file=None):
    perplexities = []
    goodperplexities = []
    total_nll = 0
    total_tokens = 0
    g = 0
    ct = 0
    if write_file is not None:
        fout = open(write_file, "w")

    # for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating PPL'):
        # prompt_input_ids = torch.LongTensor([row.prompt['tokens']]).to(device)
        prompt = row.context_string
        prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        if not (prompt_input_ids.shape[1] == 1 and prompt_input_ids[0].tolist()[0] == tokenizer.bos_token_id): # this means unconditional, prompt is BOS token (verify)
            prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)
            # print("in")
        else:
            prompt_loss = 0
            # print("out")
        # for every generation conditioned on the prompt
        generations = row.string
        for gen in generations:
            full_input_ids = tokenizer.encode(f'{prompt}{gen}', return_tensors='pt').to(device)
            full_loss = model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1]-1)
            loss = (full_loss - prompt_loss) / (full_input_ids.shape[1] - prompt_input_ids.shape[1])

            ppl = np.exp(loss.item())
            if ppl < 100:   # for sanity
                goodperplexities.append(ppl)
                # perplexities.append(ppl)
                g += 1

            if ppl < 1e4:
                perplexities.append(ppl)
            else:
                print("ppl values are weirldly large. Check for errors")
                print(f"\n########\n{gen}\n########\n")

            total_nll += (full_loss - prompt_loss).item()
            total_tokens += (full_input_ids.shape[1] - prompt_input_ids.shape[1])
            # print(full_input_ids[0], prompt_input_ids[0])
            # print(full_loss, prompt_loss)
            # input()
            if write_file is not None:
                fout.write(f"{ppl}, {(full_loss - prompt_loss).item()}, {(full_input_ids.shape[1] - prompt_input_ids.shape[1])}\n")
        
        # input("ok")
    
    print(np.nanmean(goodperplexities), len(goodperplexities), len(perplexities), g)
    return np.nanmean(perplexities), np.exp(total_nll/total_tokens)


def fluency_classify(generations_df, output_file, batch_size=32):
    from fairseq.models.roberta import RobertaModel
    from fairseq.data.data_utils import collate_tokens

    model = RobertaModel.from_pretrained(
            '/projects/tir5/users/sachink/embed-style-transfer/evaluation_models/cola_classifier_fluency/',
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path='./cola-bin'
        )
    model.cuda()

    def label_fn(label):
        return model.task.label_dictionary.string(
            [label + model.task.target_dictionary.nspecial]
        )
    
    def predict_batch(batch):
        batch = collate_tokens([model.task.source_dictionary.encode_line("<s> " + sd + " </s>", append_eos=False) for sd in batch], pad_idx=1)
        batch = batch[:, :512]

        with torch.no_grad():
            predictions = model.predict('sentence_classification_head', batch.long())
            # prediction_probs = [torch.exp(x).max(axis=0)[0].item() for x in predictions]
            prediction_labels = [label_fn(x.argmax(axis=0).item()) for x in predictions]
        
        return prediction_labels
            
    batch = []
    all_prediction_labels = []
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating CoLA fluency'):
        prompt = row.prompt['text']
        generations = [gen['text'] for gen in row['generations']]
        for j, gen in enumerate(generations):
            batch.append(model.bpe.encode(f'{prompt}{gen}'))
            if len(batch) == batch_size:
                prediction_labels = predict_batch(batch)
                all_prediction_labels += prediction_labels
                batch = []
        
        if len(batch) != 0:
            prediction_labels = predict_batch(batch)
            all_prediction_labels += prediction_labels
            batch = []
    
    with open(output_file, "w") as fout:
        fout.write("\n".join(all_prediction_labels))

    accuracy = np.array(all_prediction_labels) == "acceptable"
    accuracy = np.nanmean(accuracy.astype("float32"))
    return accuracy


def distinctness(generations_df):
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across generations for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating dist-n'):
        generations = row['string']
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in generations:
            o = gen.split(' ')
            # o = [str(tok) for tok in gen]
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i+1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i+1] + '_' + o[i+2])
        dist1.append(len(unigrams) / total_words)
        dist2.append(len(bigrams) / total_words)
        dist3.append(len(trigrams) / total_words)
    
    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)

def compute_bleu(generations_df):
    from sacrebleu.metrics import BLEU

    sys = []
    # refs = []
    refs = [[]] # nope?
    for i, row in generations_df.iterrows():
        sys.extend(row['string'])
        for _ in row['string']: # assumes >=1 references
            refs[0].append(row['gold_string'])

    bleu = BLEU()
    score = bleu.corpus_score(sys, refs)

    return score

def compute_bertscore(generations_df):
    from bert_score import score

    sys = []
    refs = []
    for i, row in generations_df.iterrows():
        sys.extend(row['string'])
        for _ in row['string']: # assumes 1 references
            refs.append(row['gold_string'])

    P, R, F1 = score(sys, refs, lang="en", verbose=False)

    return P.mean().item(), R.mean().item(), F1.mean().item()

def self_bleu(generations_df, n_sample=1000):

    # import spacy
    random.seed(0)
    # nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
    # nlp.add_pipe(nlp.create_pipe('sentencizer'))

    smoothing_function = SmoothingFunction().method1
    all_sentences = []
    for i, row in generations_df.iterrows():
        # gens = row['tokens']
        gens = [[str(token) for token in tokens] for tokens in row['tokens']]# for gen in row['generations']] {'prompt':"", tokens: [[1,2,3], [3,4,5], [5,6,7], ....]}
        all_sentences += gens
    
    pool = Pool(processes=os.cpu_count())
    bleu_scores = []
    for n_gram in range(1, 6):

        if n_gram == 1:
            weights = (1.0, 0, 0, 0)
        elif n_gram == 2:
            weights = (0.5, 0.5, 0, 0)
        elif n_gram == 3:
            weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
        elif n_gram == 4:
            weights = (0.25, 0.25, 0.25, 0.25)
        elif n_gram == 5:
            weights = (0.2, 0.2, 0.2, 0.2, 0.2)
        else:
            raise ValueError
        bleu_scores.append(
            list(tqdm(
                pool.imap_unordered(
                    partial(bleu_i, weights, all_sentences, smoothing_function),
                    random.sample(range(len(all_sentences)), min(n_sample, len(all_sentences)))),
                total=min(n_sample, len(all_sentences)),
                smoothing=0.0,
                desc=f"bleu-{n_gram}")))
        # print(f"\n\nbleu-{n_gram} = {sum(bleu_scores[n_gram - 1]) / n_sample}")
    
    pool.close()
    pool.join()

    bleus = []
    for n_gram in range(5):
        bleus.append(sum(bleu_scores[n_gram]) / n_sample)
        # print(f"bleu-{n_gram + 1} = {sum(bleu_scores[n_gram]) / n_sample}")
    
    return bleus

    # if args.logto:
    #     with open(args.logto, 'a') as fout:
    #         print(f"{os.path.basename(args.file)}", end='\t', file=fout)
    #         for n_gram in range(5):
    #             print(f"{sum(bleu_scores[n_gram]) / n_sample}", end='\t', file=fout)
    #         print(file=fout)


def self_bleu2(generations_df, n_sample=100):

    # import spacy
    random.seed(0)
    smoothing_function = SmoothingFunction().method1
    # nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
    # nlp.add_pipe(nlp.create_pipe('sentencizer'))
    all_bleus = [[] for _ in range(3)]
    for i, row in generations_df.iterrows():
        # all_sentences = []
        all_sentences = row['tokens']# for gen in row['generations']]
        # all_sentences += gens
        
        pool = Pool(processes=os.cpu_count())
        bleu_scores = []
        for i in range(3):
            n_gram = i+3
            if n_gram == 1:
                weights = (1.0, 0, 0, 0)
            elif n_gram == 2:
                weights = (0.5, 0.5, 0, 0)
            elif n_gram == 3:
                weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
            elif n_gram == 4:
                weights = (0.25, 0.25, 0.25, 0.25)
            elif n_gram == 5:
                weights = (0.2, 0.2, 0.2, 0.2, 0.2)
            else:
                raise ValueError
            bleu_scores.append(
                list(tqdm(
                    pool.imap_unordered(
                        partial(bleu_i, weights, all_sentences, smoothing_function),
                        random.sample(range(len(all_sentences)), min(n_sample, len(all_sentences)))),
                    total=min(n_sample, len(all_sentences)),
                    smoothing=0.0,
                    desc=f"bleu-{n_gram}")))
            # print(f"\n\nbleu-{n_gram} = {sum(bleu_scores[n_gram - 1]) / n_sample}")
        
        pool.close()
        pool.join()

        for i in range(3):
            all_bleus[i].append(sum(bleu_scores[i]) / n_sample)
            # print(f"bleu-{n_gram + 1} = {sum(bleu_scores[n_gram]) / n_sample}")
    all_bleus = [np.nanmean(bleu) for bleu in all_bleus]
    return all_bleus

    # if args.logto:
    #     with open(args.logto, 'a') as fout:
    #         print(f"{os.path.basename(args.file)}", end='\t', file=fout)
    #         for n_gram in range(5):
    #             print(f"{sum(bleu_scores[n_gram]) / n_sample}", end='\t', file=fout)
    #         print(file=fout)


def zipf_coefficient(generations_df, N=5000):
    cnt = Counter()
    
    for i, row in generations_df.iterrows():
        generations = row['tokens']# for gen in row['generations']]
        for gen in generations:
            cnt.update(gen)

    xs = np.arange(1, min(len(cnt), N)+1)
    ys = np.array(sorted(cnt.values(), key=operator.neg)[:N])
    s, b, r, p, std = stats.linregress(np.log(xs), np.log(ys))
    return -s, -r, p


def mauve_score(generations_df):
    import mauve 

    machine_text = []
    human_text = []
    for i, row in generations_df.iterrows():
        prompt = row['context_string']
        for output in row['string']:
            machine_text.append(f'{prompt}{output}')
        human_text.append(f'{prompt}{row["gold_string"]}')


    # call mauve.compute_mauve using raw text on GPU 0; each generation is truncated to 256 tokens
    out = mauve.compute_mauve(p_text=human_text, q_text=machine_text, device_id=0, max_text_length=256, verbose=False)
    print(out.mauve)

    return out.mauve


def repetition(generations_df, tokenizer, numbers_only=True, rep_file=None):
    SEP = tokenizer.encode(tokenizer.bos_token)[0]

    objs = []
    max_n = 90

    n_repeated_examples = 0
    total_examples = 0

    if rep_file is not None:
        fout = open(rep_file, "w")
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating repetitions'):
        generations = row['tokens'] #for gen in row['generations']]
        for gen in generations:
            total_examples += 1
            if gen[-1] == SEP:
                gen.pop(-1)
            rev_gen = list(reversed(gen))
            last_n_repeats = [0] * max_n

            for n in range(1, max_n + 1):
                n_repeat = 1
                while len(rev_gen[n*n_repeat:n*(n_repeat+1)]) == n and \
                        rev_gen[n*n_repeat:n*(n_repeat+1)] == rev_gen[:n]:
                    n_repeat += 1
                last_n_repeats[n - 1] = n_repeat
            max_repeated_n = max(range(max_n), key=lambda x: last_n_repeats[x])

            if last_n_repeats[max_repeated_n] > 1 and (max_repeated_n+1 >= 3 or last_n_repeats[max_repeated_n] > 50):
                repetition = {
                    'repeated_phrase': list(reversed(rev_gen[:max_repeated_n + 1])),
                    'repeated_times': last_n_repeats[max_repeated_n],
                    'repeated_phrase_length': max_repeated_n + 1,
                }
                n_repeated_examples += 1
            else:
                repetition = {}
            
            if rep_file is not None:
                json.dump(repetition, fout)
                fout.write("\n")
    
    if rep_file is not None:
        fout.close()

    return n_repeated_examples*1.0/total_examples

    # if not numbers_only:
    #     print("filename\tnumber of repeating examples")
    #     print(f"{os.path.basename(args.file)}\t{n_repeated_examples}")
    # if args.output:
    #     output_filename = os.path.join(os.path.dirname(args.file), "repetition_" + os.path.basename(args.file))
    #     with open(output_filename, 'w+') as fout:
    #         for obj in objs:
    #             print(json.dumps(obj), file=fout)


def compute_intctr(generations_df, ctr_label_idx): # cardiffnlp/twitter-roberta-base-sentiment
    ctr_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(ctr_model_name)
    config = AutoConfig.from_pretrained(ctr_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(ctr_model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')

    machine_text = []
    for i, row in generations_df.iterrows():
        prompt = row['context_string']
        for output in row['string']:
            machine_text.append(f'{prompt}{output}')

    acc_list = []
    for text in machine_text:
        encoded_input = tokenizer(text, return_tensors='pt')
        if torch.cuda.is_available():
            encoded_input = encoded_input.to('cuda')
        output = model(**encoded_input)
        output.logits[:, 1] = -float('inf') # cardiffnlp/twitter-roberta-base-sentiment's label idx 1 is neutral, which we should ignore
        argmax_label_list = output.logits.argmax(dim=-1).view(-1).tolist()
        for _l in argmax_label_list:
            if _l == ctr_label_idx:
                acc_list.append(1)
            else:
                acc_list.append(0)
    acc = sum(acc_list) / len(acc_list)
    return acc


def compute_extctr1(generations_df, ctr_label_idx): # 
    ctr_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    if ctr_label_idx == 2:
        ctr_label_idx = 1 # cardiffnlp/twitter-roberta-base-sentiment's label idx 2 is positive sentiment (1 in other models)

    tokenizer = AutoTokenizer.from_pretrained(ctr_model_name)
    config = AutoConfig.from_pretrained(ctr_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(ctr_model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')

    machine_text = []
    for i, row in generations_df.iterrows():
        prompt = row['context_string']
        for output in row['string']:
            machine_text.append(f'{prompt}{output}')

    acc_list = []
    for text in machine_text:
        encoded_input = tokenizer(text, return_tensors='pt')
        if torch.cuda.is_available():
            encoded_input = encoded_input.to('cuda')
        output = model(**encoded_input)
        argmax_label_list = output.logits.argmax(dim=-1).view(-1).tolist()
        for _l in argmax_label_list:
            if _l == ctr_label_idx:
                acc_list.append(1)
            else:
                acc_list.append(0)
    acc = sum(acc_list) / len(acc_list)
    return acc


def compute_extctr2(generations_df, ctr_label_idx): # 
    ctr_model_name = "textattack/bert-base-uncased-yelp-polarity"
    if ctr_label_idx == 2:
        ctr_label_idx = 1 # cardiffnlp/twitter-roberta-base-sentiment's label idx 2 is positive sentiment (1 in other models)

    tokenizer = AutoTokenizer.from_pretrained(ctr_model_name)
    config = AutoConfig.from_pretrained(ctr_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(ctr_model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')

    machine_text = []
    for i, row in generations_df.iterrows():
        prompt = row['context_string']
        for output in row['string']:
            machine_text.append(f'{prompt}{output}')

    acc_list = []
    for text in machine_text:
        encoded_input = tokenizer(text, return_tensors='pt')
        if torch.cuda.is_available():
            encoded_input = encoded_input.to('cuda')
        output = model(**encoded_input)
        argmax_label_list = output.logits.argmax(dim=-1).view(-1).tolist()
        for _l in argmax_label_list:
            if _l == ctr_label_idx:
                acc_list.append(1)
            else:
                acc_list.append(0)
    acc = sum(acc_list) / len(acc_list)
    return acc


def compute_extctr3(generations_df, ctr_label_idx): # 
    ctr_model_name = "siebert/sentiment-roberta-large-english"
    if ctr_label_idx == 2:
        ctr_label_idx = 1 # cardiffnlp/twitter-roberta-base-sentiment's label idx 2 is positive sentiment (1 in other models)

    tokenizer = AutoTokenizer.from_pretrained(ctr_model_name)
    config = AutoConfig.from_pretrained(ctr_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(ctr_model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')

    machine_text = []
    for i, row in generations_df.iterrows():
        prompt = row['context_string']
        for output in row['string']:
            machine_text.append(f'{prompt}{output}')

    acc_list = []
    for text in machine_text:
        encoded_input = tokenizer(text, return_tensors='pt')
        if torch.cuda.is_available():
            encoded_input = encoded_input.to('cuda')
        output = model(**encoded_input)
        argmax_label_list = output.logits.argmax(dim=-1).view(-1).tolist()
        for _l in argmax_label_list:
            if _l == ctr_label_idx:
                acc_list.append(1)
            else:
                acc_list.append(0)
    acc = sum(acc_list) / len(acc_list)
    return acc


def compute_extctr4(generations_df, ctr_label_idx): # 
    ctr_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    tokenizer = AutoTokenizer.from_pretrained(ctr_model_name)
    config = AutoConfig.from_pretrained(ctr_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(ctr_model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')

    machine_text = []
    for i, row in generations_df.iterrows():
        prompt = row['context_string']
        for output in row['string']:
            machine_text.append(f'{prompt}{output}')

    acc_list = []
    for text in machine_text:
        encoded_input = tokenizer(text, return_tensors='pt')
        if torch.cuda.is_available():
            encoded_input = encoded_input.to('cuda')
        output = model(**encoded_input)
        output.logits[:, 1] = -float('inf') # since it's not binary
        argmax_label_list = output.logits.argmax(dim=-1).view(-1).tolist()
        for _l in argmax_label_list:
            if _l == ctr_label_idx:
                acc_list.append(1)
            else:
                acc_list.append(0)
    acc = sum(acc_list) / len(acc_list)
    return acc


def compute_extctr5(generations_df, ctr_label_idx): # 
    ctr_model_name = "finiteautomata/bertweet-base-sentiment-analysis"

    tokenizer = AutoTokenizer.from_pretrained(ctr_model_name)
    config = AutoConfig.from_pretrained(ctr_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(ctr_model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')

    machine_text = []
    for i, row in generations_df.iterrows():
        prompt = row['context_string']
        for output in row['string']:
            machine_text.append(f'{prompt}{output}')

    acc_list = []
    for text in machine_text:
        encoded_input = tokenizer(text, return_tensors='pt')
        if torch.cuda.is_available():
            encoded_input = encoded_input.to('cuda')
        output = model(**encoded_input)
        output.logits[:, 1] = -float('inf') # since it's not binary
        argmax_label_list = output.logits.argmax(dim=-1).view(-1).tolist()
        for _l in argmax_label_list:
            if _l == ctr_label_idx:
                acc_list.append(1)
            else:
                acc_list.append(0)
    acc = sum(acc_list) / len(acc_list)
    return acc

def dummy_length(generations_df):
    eval_tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    len_list = []
    for i, row in generations_df.iterrows():
        for s in row['string']:
            len_list.append(len(eval_tokenizer.encode(s)))
    return 1.0 * sum(len_list) / len(len_list)


@click.command()
@click.option('--generations_file', required=True, type=str, help='a jsonl file with generations and attribute scores')
@click.option('--output_file', required=True, type=str, help='filename to write the results to')
@click.option('--metrics', required=True, type=str, help='which metrics to compute, write comma separeted, ppl-mid,ppl-big,cola,self-bleu,zipf,repetition,dist-n')
@click.option('--extra', required=False, type=str, help='extra params')
def main(generations_file, output_file, metrics, extra):
    assert os.path.exists(generations_file)
    output_dir = Path(os.path.dirname(generations_file))
    generations_df = pd.read_json(generations_file, lines=True)
    
    metricset = set(metrics.strip().split(",")) # cannot use lower here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### calculate quality metrics

    # Fluency
    fo = open(output_dir / output_file, 'w') #creating the file
    fo.close()
    
    # print(metrics)
    if "ppl" in metrics:
        allmetrics = metrics.split(",")
        for metric in metricset:
            if "ppl" in metric:
                eval_modelname = metric.split("#")[1]
                print(f'computing {eval_modelname} ppl')
                eval_model = AutoModelForCausalLM.from_pretrained(eval_modelname).to(device)
                eval_tokenizer = AutoTokenizer.from_pretrained(eval_modelname)
                torch.cuda.empty_cache()
                with torch.no_grad():
                    ppl, total_ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device, write_file=output_dir / (output_file+".ppl-"+eval_modelname.replace("/", "-")))

                # write output results
                with open(output_dir / output_file, 'a') as fo:
                    fo.write(f'{eval_modelname} perplexity, {eval_modelname} total perplexity = {ppl}, {total_ppl}\n')
                    print(f'{eval_modelname} perplexity, {eval_modelname} total perplexity = {ppl}, {total_ppl}\n')
    
    if "mauve" in metricset:
        print("computing mauve")
        mauavescore = mauve_score(generations_df)

        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mauve score = {mauavescore}\n')
            print(f'mauve score = {mauavescore}\n')

    #cola
    if "cola" in metricset:
        print("computing fluency (cola)")
        cola_accuracy = fluency_classify(generations_df, output_file=output_dir / (output_file+".cola"))
        
        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'cola acceptability accuracy = {cola_accuracy}\n')
            print(cola_accuracy)

    ### calculate diversity
    # dist-n
    if "dist-n" in metricset:
        dist1, dist2, dist3 = distinctness(generations_df)
        
        # write output results
        with open(output_dir / output_file, 'a') as fo:
            for i, dist_n in enumerate([dist1, dist2, dist3]):
                fo.write(f'dist-{i+1} = {dist_n}\n')
                print(f'dist-{i+1} = {dist_n}')

    # self-bleu
    if "self-bleu" in metricset:
        bleus = self_bleu(generations_df)
        with open(output_dir / output_file, 'a') as fo:
            for i, bleu in enumerate(bleus):
                fo.write(f'bleu-{i+1} = {bleu}\n')
                print(f'bleu-{i+1} = {bleu}')
    
    # self-bleu
    if "self-bleu2" in metricset:
        bleus = self_bleu2(generations_df)
        with open(output_dir / output_file, 'a') as fo:
            for i, bleu in enumerate(bleus):
                fo.write(f'bleu2-{i+3} = {bleu}\n')
                print(f'bleu2-{i+3} = {bleu}')

    # zipf-coefficient
    if "zipf" in metricset:
        s, r, p = zipf_coefficient(generations_df)
        
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'zipf: s={s}, r={r}, p={p}\n')
            print(f'zipf: s={s}, r={r}, p={p}')

    # repetition
    if "repetition" in metricset:
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
        rep_rate = repetition(generations_df, eval_tokenizer, rep_file=output_dir / (output_file+".repetitions"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'repetition_rate: {rep_rate}\n')
            print(f'repetition_rate: {rep_rate}')

    # bleu 
    if "bleu" in metricset:
        bleuoutput = compute_bleu(generations_df)
        
        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'BLEU: {bleuoutput}\n')
            print(f'BLEU: {bleuoutput}')
    
    # bertscore
    if "bertscore" in metricset:
        p, r, f1 = compute_bertscore(generations_df)
        
        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'BertScore (P/R/F1) = {p}/{r}/{f1}\n')
            print(f'BertScore (P/R/F1) = {p}/{r}/{f1}')

    # intctr
    if "intctr" in metricset: 
        assert 'ctridx' in generations_file
        if 'ctridx0' in generations_file:
            acc = compute_intctr(generations_df, 0)
        elif 'ctridx2' in generations_file:
            acc = compute_intctr(generations_df, 2)
        else:
            raise ValueError("check ctridx")
        
        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'Internal classifier acc = {acc}\n')
            print(f'Internal classifier acc = {acc}')

    # extctr1
    if "extctr1" in metricset: 
        assert 'ctridx' in generations_file
        if 'ctridx0' in generations_file:
            acc = compute_extctr1(generations_df, 0)
        elif 'ctridx2' in generations_file:
            acc = compute_extctr1(generations_df, 2)
        else:
            raise ValueError("check ctridx")
        
        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'External classifier 1 acc = {acc}\n')
            print(f'External classifier 1 acc = {acc}')

    # extctr2
    if "extctr2" in metricset: 
        assert 'ctridx' in generations_file
        if 'ctridx0' in generations_file:
            acc = compute_extctr2(generations_df, 0)
        elif 'ctridx2' in generations_file:
            acc = compute_extctr2(generations_df, 2)
        else:
            raise ValueError("check ctridx")
        
        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'External classifier 2 acc = {acc}\n')
            print(f'External classifier 2 acc = {acc}')

    # extctr3
    if "extctr3" in metricset: 
        assert 'ctridx' in generations_file
        if 'ctridx0' in generations_file:
            acc = compute_extctr3(generations_df, 0)
        elif 'ctridx2' in generations_file:
            acc = compute_extctr3(generations_df, 2)
        else:
            raise ValueError("check ctridx")
        
        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'External classifier 3 acc = {acc}\n')
            print(f'External classifier 3 acc = {acc}')

    # extctr4
    if "extctr4" in metricset: 
        assert 'ctridx' in generations_file
        if 'ctridx0' in generations_file:
            acc = compute_extctr4(generations_df, 0)
        elif 'ctridx2' in generations_file:
            acc = compute_extctr4(generations_df, 2)
        else:
            raise ValueError("check ctridx")
        
        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'External classifier 4 acc = {acc}\n')
            print(f'External classifier 4 acc = {acc}')

    # extctr5
    if "extctr5" in metricset: 
        assert 'ctridx' in generations_file
        if 'ctridx0' in generations_file:
            acc = compute_extctr5(generations_df, 0)
        elif 'ctridx2' in generations_file:
            acc = compute_extctr5(generations_df, 2)
        else:
            raise ValueError("check ctridx")
        
        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'External classifier 5 acc = {acc}\n')
            print(f'External classifier 5 acc = {acc}')

    # simply the length of the generations (as a sanity check)
    if "dummylen" in metricset:
        print("computing dummy length")
        dummylen = dummy_length(generations_df)

        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'dummy length = {dummylen}\n')
            print(f'dummy length = {dummylen}')

if __name__ == '__main__':
    main()