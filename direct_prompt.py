"""
This module is to experiment for analysis of "How is the performance of In-context Learning?"
"""

import logging
import json
import pickle
import argparse

import tqdm
import wandb
import pandas as pd
import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration, AutoModelForSeq2SeqLM
from few_shot_prompt_utility import format_prompt_from_demo_pairs, prompt_T5, evaluate_response_summaries

wandb.login(key='3138e1b24deb278ed045d0dedb39511d3a96245b')

# set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='google/mt5-xl', help='the name of the model')
parser.add_argument('-k', type=int, default=1, help='the number of few-shot examples')
parser.add_argument('--demonstration_file', type=str, default=None, help='the pre-generated demonstration file')
parser.add_argument('--dataset', type=str, default='samsum', help='the dataset to evaluate on')
# parse the arguments
args = parser.parse_args()

# start a new wandb run to track this script
wandb.init(
        project="In-context-learning for Dialogue Summarization",
        # track hyperparameters and run metadata
        config={
                'model_type': args.model,
                'k': args.k,
                'dataset': args.dataset,
                },
        group='performance_in_context_learning',
        job_type='evaluation'
        )

# load the demonstration pickle file
with open(args.demonstration_file, 'rb') as f:
    run_id2demo_pairs = pickle.load(f)

run_id2prompts, run_id2gold_summaries = format_prompt_from_demo_pairs(run_id2demo_pairs, args.model)

logging.info("load the model {}".format(args.model))
if args.model in ['google/mt5-xl', 'google/mt5-base']:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = MT5ForConditionalGeneration.from_pretrained(args.model).to(DEVICE)
elif args.model == 'google/mt5-xxl':
    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-xxl", device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
else:
    raise NotImplementedError('The model is not implemented yet.')

logging.info("Start to prompt the model")
run_id2pred_summaries = {}
for run_id, prompts in run_id2prompts.items():
    logging.info("Prompting the model for Run {}".format(run_id))
    run_id2pred_summaries[run_id] = []
    for prompt in tqdm.tqdm(prompts):
        if args.model in ['google/mt5-xl', 'google/mt5-base', 'google/mt5-xxl']:
            response = prompt_T5(model, tokenizer, prompt)
        else:
            raise NotImplementedError('The model is not implemented yet.')
        run_id2pred_summaries[run_id].append(response)


logging.info("Start to evaluate the performance")
summary_table, summary_text_table = evaluate_response_summaries(run_id2pred_summaries, run_id2gold_summaries,
                                                                run_id2prompts)

# save the summary table to wandb
summary_table = pd.DataFrame(summary_table)
summary_text_table = pd.DataFrame(summary_text_table)
summary_table = wandb.Table(dataframe=summary_table)
summary_text_table = wandb.Table(dataframe=summary_text_table)
wandb.log({"Evaluation metrics Table": summary_table})
wandb.log({"Summaries Table": summary_text_table})
wandb.finish()
