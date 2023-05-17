"""
This module is to experiment for analysis of "How is the performance of In-context Learning?"
"""


import logging
import json
import pickle
import argparse

import tqdm
import wandb
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration
from rouge_score import rouge_scorer
from few_shot_prompt_utility import format_prompt_from_demo_pairs, prompt_T5


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
wandb_session = wandb.init(
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
else:
    raise NotImplementedError('The model is not implemented yet.')


logging.info("Start to prompt the model")
run_id2pred_summaries = {}
for run_id, prompts in run_id2prompts.items():
    logging.info("Prompting the model for Run {}".format(run_id))
    run_id2pred_summaries[run_id] = []
    for prompt in tqdm.tqdm(prompts):
        if args.model in ['google/mt5-xl', 'google/mt5-base']:
            response = prompt_T5(model, tokenizer, prompt)
        else:
            raise NotImplementedError('The model is not implemented yet.')
        run_id2pred_summaries[run_id].append(response)


# evaluate the model
summary_table = []
summary_text_table = []
for run_id, pred_summaries in run_id2pred_summaries.items():
    scores_in_run = []
    logging.info("Evaluating the model for Run {}".format(run_id))
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    for i, pred_summary in enumerate(pred_summaries):
        gold_summary = run_id2gold_summaries[run_id][i]
        scores = scorer.score(pred_summary, gold_summary)
        scores_in_run.append(scores)
        summary_text_row = {
            'run_id': run_id,
            'prompt': run_id2prompts[run_id][i],
            'pred_summary': pred_summary,
            'gold_summary': gold_summary
        }
        summary_text_table.append(summary_text_row)
    summary_row = {
            'run_id': run_id,
            'rouge_1_precision': np.mean([score['rouge1'].precision for score in scores_in_run]),
            'rouge_1_recall': np.mean([score['rouge1'].recall for score in scores_in_run]),
            'rouge_1_fmeasure': np.mean([score['rouge1'].fmeasure for score in scores_in_run]),
            'rouge_2_precision': np.mean([score['rouge2'].precision for score in scores_in_run]),
            'rouge_2_recall': np.mean([score['rouge2'].recall for score in scores_in_run]),
            'rouge_2_fmeasure': np.mean([score['rouge2'].fmeasure for score in scores_in_run]),
            'rouge_L_precision': np.mean([score['rougeL'].precision for score in scores_in_run]),
            'rouge_L_recall': np.mean([score['rougeL'].recall for score in scores_in_run]),
            'rouge_L_fmeasure': np.mean([score['rougeL'].fmeasure for score in scores_in_run]),
            }
    summary_table.append(summary_row)

# save the summary table to wandb
summary_table = pd.DataFrame(summary_table)
summary_text_table = pd.DataFrame(summary_text_table)
summary_table = wandb.Table(dataframe=summary_table)
summary_text_table = wandb.Table(dataframe=summary_text_table)
wandb_session.log({"Evaluation metrics Table": summary_table})
wandb_session.log({"Summaries Table": summary_text_table})
wandb_session.finish()
