"""
Make inference using LLM on dialogue training dataset through random sampling without control signals.
Few-shot examples are randomly sampled.
The results are evaluated using ROUGE scores.
The response and the scores are exported to local csv files.

TODO:
- add control signals to the inference process
- Understand beam search

"""


import logging
import sys
from datetime import datetime


import pandas as pd
import torch
from datasets import load_dataset
from rouge import Rouge
from tqdm import tqdm
from transformers import AutoTokenizer, MT5ForConditionalGeneration


# hyper-parameters
MODEL = 'google/mt5-xl'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_NUM = 1000
FEW_SHOT_NUM = 1


# remove any special character in MODEL as the prefix of csv file
csv_prefix = MODEL.replace('/', '_')
csv_file_name = '{}_{}shot.csv'.format(csv_prefix, FEW_SHOT_NUM)


# set up log files
logging.basicConfig(
        level=logging.INFO,  # otherwise huggingface has many debug logs
        handlers=[
                logging.FileHandler("{}.log".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))),
                logging.StreamHandler(sys.stdout)
                ]
        )


# set up model
logging.info('Loading {} to Device {}'.format(MODEL, DEVICE))
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = MT5ForConditionalGeneration.from_pretrained(MODEL).to(DEVICE)


# load dataset
logging.info('Loading dataset')
dataset = load_dataset("samsum")


# randomly sample 1000 dialogues from the dataset
df = dataset['train'].to_pandas()
train_data = df.sample(SAMPLE_NUM)
# log the index of the sampled dialogues
logging.info('Sampled dialogues: {}'.format(train_data.id.tolist()))


# generate few-shot examples
few_shot_samples = []
for index, row in train_data.iterrows():
    candidates = df.drop(index)
    train_sample = candidates.sample(FEW_SHOT_NUM)
    few_shot_samples.append(train_sample)
# log the index of the sampled few-shot examples
logging.info('Sampled few-shot examples: {}'.format([sample.id.tolist() for sample in few_shot_samples]))


# prepare the prompt and append the prompt to train_data DataFrame
counter = 0
prompts = []
for _, row in train_data.iterrows():
    train_sample = few_shot_samples[counter]
    train_str = ""
    for _, sample in train_sample.iterrows():
        train_str += 'Conversation: ' + sample['dialogue'].strip().replace("\n", " ").replace("\r",
                                                                                              " ") + '\nsummary: ' + \
                     sample['summary']

    prompt = "summarize the conversation.\n" + train_str + "\n" + 'Conversation: ' + row['dialogue'] + '\nsummary: '
    prompts.append(prompt)

    counter += 1

train_data['processed_dialogue'] = prompts


# make inference without SAP
responses = []

prog_bar = tqdm(total=len(train_data))
for _, row in train_data.iterrows():
    prog_bar.update(1)
    prompt = row['processed_dialogue']
    sap_counter = 0
    # tokenize dialogue data
    X = tokenizer.encode(prompt + '<extra_id_0>', return_tensors="pt").to(DEVICE)
    # Summarize
    y_ids = model.generate(X,
                           num_beams=4,
                           no_repeat_ngram_size=2,
                           early_stopping=True)
    y = tokenizer.decode(y_ids[0], skip_special_tokens=True)

    responses.append(y)


# parse the responses
responses = [x.strip().split('<extra_id_0>')[-1] for x in responses]


logging.info('Evaluating the responses in ROUGE scores')
rouge = Rouge()
try:
    rouge_scores = rouge.get_scores(responses, train_data['summary'].tolist())
except ValueError:  # raised when hypothsis is empty
    # create a DataFrame with two columns: responses and train_data['summary'].tolist()
    df = pd.DataFrame(list(zip(responses, train_data['summary'].tolist())), columns=['response', 'summary'])
    logging.info('empty hypothesis, csv is exported without rouge score')
    df.to_csv(csv_file_name)
    # end the program
    sys.exit(0)

logging.info('Exporting the responses and the scores to local csv files')
df_results = []
for i in range(len(train_data)):
    session = train_data.iloc[i]
    response = responses[i]
    d = {
            'session_id': session.index,
            'gold_summary': session['summary'],
            'response_summary': response,
            'rouge_1_f': rouge_scores[i]['rouge-1']['f'],
            'rouge_2_f': rouge_scores[i]['rouge-2']['f'],
            'rouge_l_f': rouge_scores[i]['rouge-l']['f'],
            'rouge_1_p': rouge_scores[i]['rouge-1']['p'],
            'rouge_2_p': rouge_scores[i]['rouge-2']['p'],
            'rouge_l_p': rouge_scores[i]['rouge-l']['p'],
            'rouge_1_r': rouge_scores[i]['rouge-1']['r'],
            'rouge_2_r': rouge_scores[i]['rouge-2']['r'],
            'rouge_l_r': rouge_scores[i]['rouge-l']['r'],
            'scores': rouge_scores[i],
            #         'keywords': session['keywords'],
            #         'missing_keywords': [x for x in session['keywords'] if x not in predictions[i].lower()]
            }
    df_results.append(d)

pd.DataFrame(df_results).to_csv(csv_file_name)
