"""
This module is to calculate the perplexity of the test data fetched from wandb.
"""


import sys
import json
from io import StringIO
import pandas as pd
from datetime import datetime
import argparse
import wandb
import logging
from tqdm.auto import tqdm
import torch
try:
    from transformers import AutoTokenizer, MT5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoModelForCausalLM, \
        LlamaTokenizer, LlamaForCausalLM
except ImportError:  # old huggingface does not have LlmaTimeForCausalLM
    from transformers import AutoTokenizer, MT5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from few_shot_prompt_utility import format_prompt_from_demo_pairs, prompt_llm, evaluate_response_summaries, \
    generate_tf_idf_keywords
from datasets import load_dataset


# set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='the name of the model to load from HuggingFace')
parser.add_argument('--run_name', type=str, help='the name of run to fetch the test data from wandb')
# below are arguments for logging purpose
parser.add_argument('-k', type=int, default=1, help='the number of few-shot examples')
parser.add_argument('--dataset', type=str, default='samsum', help='the dataset to evaluate on')
parser.add_argument('--keywords', type=str, default=None, choices=['tfidf'], help='the types of keywords to use')
parser.add_argument('--keyword_num', type=int, default=None, help='the number of keywords to use')
# parse the arguments
args = parser.parse_args()


wandb.init(
        project="In-context-learning for Dialogue Summarization",
        # track hyperparameters and run metadata
        config={
                'model_type': args.model,
                'k': args.k,
                'dataset': args.dataset,
                'keywords': args.keywords,
                'keyword_num': args.keyword_num,
                'run_name': args.run_name,
                },
        group='performance_in_context_learning',
        job_type='perplexity_evaluation'
        )

# set up log files
logging.basicConfig(
        level=logging.INFO,  # otherwise huggingface has many debug logs
        handlers=[
                logging.FileHandler("{}.log".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))),
                logging.StreamHandler(sys.stdout)
                ]
        )


# connect to wandb and get list of runs
api_helper = wandb.Api(api_key='3138e1b24deb278ed045d0dedb39511d3a96245b')
runs = list(api_helper.runs(path='yuting_fyp/In-context-learning for Dialogue Summarization',
                     per_page=1000))
logging.info(f'Number of runs fetched from WanDB: {len(runs)}')


# iterate over runs to get the artifects
dfs = []
for run in tqdm(runs):
    if 'complete' not in run.tags or getattr(run, 'job_type', '') != 'evaluation':  # skip the unfinalized model
        continue
    if run.name != args.run_name:  # skip the irrelevant runs
        continue
    files = run.files()
    metric_file = [f for f in files if 'Summaries Table' in getattr(f, 'name', '')]
    assert len(metric_file) == 1
    metric_file = metric_file[0]
    f = metric_file.download(root='wandb', replace=True)
    summaries = json.load(f)
    summaries = json.dumps(summaries)
    df = pd.read_json(StringIO(summaries), orient='split')
    dfs.append(df)
assert len(dfs) >= 1
logging.info(f'Number of tables fetched from WanDB: {len(dfs)}')


df = pd.concat(dfs, axis=0)
logging.info(f'shape of the dataframe: {df.shape}')

logging.info("load the model {}".format(args.model))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_count = torch.cuda.device_count()
logging.info('device count = {}'.format(device_count))

# load the model
if args.model in ['google/mt5-xl', 'google/mt5-base']:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = MT5ForConditionalGeneration.from_pretrained(args.model).to(DEVICE)
elif args.model in ['google/mt5-xxl', 'google/flan-t5-xl']:
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
elif 'llama' in args.model or 'alpaca' in args.model:
    model = LlamaForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16)
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map='auto')
    if 'opt' in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        logging.info("device_map: {}".format(model.hf_device_map))
    except AttributeError:
        # send model to GPU
        model = model.to(DEVICE)

# group the df by run_id, and iterate the group to get average perplexity
grouped = df.groupby('run_id')
perplexities = []
perplexities_df = []
for run_id, group_df in tqdm(grouped):
    # iterate the rows of group_df
    for _, row in tqdm(group_df.iterrows()):
        prompt_text = row['prompt']
        gold_summary = row['gold_summary']

        # tokenize to get input_ids
        prompt_text_ids = tokenizer.encode(prompt_text, return_tensors="pt")
        summary_ids = tokenizer.encode(gold_summary, return_tensors="pt")
        eos_token_id = torch.tensor([[tokenizer.eos_token_id]])
        if device_count > 1:
            input_ids = torch.cat([prompt_text_ids, summary_ids, eos_token_id], dim=-1).to(device_count - 1)
        else:
            input_ids = torch.cat([prompt_text_ids, summary_ids, eos_token_id], dim=-1).to(DEVICE)
        target_ids = input_ids.clone()
        # mask out the elements from prompt_text_ids in target_ids to -100 so that context is not counted in loss
        target_ids[:, :prompt_text_ids.shape[-1]] = -100

        # get the perplexity
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
            perplexity = torch.exp(neg_log_likelihood)
        # append row to perplexities_df
        perplexities_df.append({'run_id': run_id,
                                'prompt': prompt_text,
                                'gold_summary': gold_summary,
                                'perplexity': perplexity.item()})

# convert perplexities_df to df
perplexities_df = pd.DataFrame(perplexities_df)
perplexities_df = wandb.Table(dataframe=perplexities_df)
wandb.log({"Perplexity Table": perplexities_df})
wandb.finish()
