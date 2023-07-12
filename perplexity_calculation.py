"""
This module is to calculate the perplexity of the test data fetched from wandb, using GPT-2.
GPT3 can be used as well.
"""


import sys
import json
from io import StringIO
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from evaluate import load
import wandb
import logging
from tqdm.auto import tqdm
import torch
from few_shot_prompt_utility import format_prompt_from_demo_pairs, prompt_llm, evaluate_response_summaries, \
    generate_tf_idf_keywords
from datasets import load_dataset


# set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, help='the name of run to fetch the test data from wandb')

# parse the arguments
args = parser.parse_args()


# wandb.init(
#         project="In-context-learning for Dialogue Summarization",
#         # track hyperparameters and run metadata
#         config={
#                 'model_type': args.model,
#                 'k': args.k,
#                 'dataset': args.dataset,
#                 'keywords': args.keywords,
#                 'keyword_num': args.keyword_num,
#                 'run_name': args.run_name,
#                 },
#         group='performance_in_context_learning',
#         job_type='perplexity_evaluation'
#         )

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

# load perplexity
perplexity = load("perplexity", module_type="metric")


# group the df by run_id, and iterate the group to get average perplexity
grouped = df.groupby('run_id')
perplexities = []
for run_id, group_df in tqdm(grouped):
    # get only the non-empty summaries
    group_df = group_df[group_df['pred_summary'] != '']
    # convert group_df['pred_summary'] to list
    summaries = group_df['pred_summary'].tolist()
    # compute perplexity and print the average
    perplexity_value = perplexity.compute(predictions=summaries, model_id='gpt2')

    perplexities.append(perplexity_value['mean_perplexity'])

# print the average perplexity
print(f'Average perplexity: {np.mean(perplexities)}')


# # convert perplexities_df to df
# perplexities_df = pd.DataFrame(perplexities_df)
# perplexities_df = wandb.Table(dataframe=perplexities_df)
# # wandb.log({"Perplexity Table": perplexities_df})
# wandb.finish()
