"""
Make inference using LLM on dialogue training dataset through random sampling with/without control signals.
Few-shot examples are randomly sampled.
Types of control signals:
    1. no control signal
    2. control signal: TF-IDF
The results are evaluated using ROUGE scores.
The response and the scores are exported to local csv files.

TODO:
- Understand beam search
- check the correctness of the inference process (e.g., compare the one from Jupyter notebook and this one)
- Add random selections of few-shot examples to verify that through API
- (minor) export to csv incrementally
"""

import logging
import sys
import re
from datetime import datetime

import pandas as pd
import torch
from datasets import load_dataset
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoTokenizer, MT5ForConditionalGeneration


# set up log files
logging.basicConfig(
        level=logging.INFO,  # otherwise huggingface has many debug logs
        handlers=[
                logging.FileHandler("{}.log".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))),
                logging.StreamHandler(sys.stdout)
                ]
        )


# hyper-parameters
MODEL = 'google/mt5-xl'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_NUM = -1
FEW_SHOT_NUM = 3
IS_ADD_CONTROL_SIGNAL = False
# control signal related parameters
IS_UPPER_BOND = True
NUMBER_OF_KEYWORDS = 3


# log sample number and few shot number
logging.info('Sample number: {}'.format(SAMPLE_NUM))
logging.info('Few-shot number: {}'.format(FEW_SHOT_NUM))

# some logging settings
IS_LOG_PROMPTS = True

# remove any special character in MODEL as the prefix of csv file
csv_prefix = MODEL.replace('/', '_')
csv_file_name = '{}_{}shot_{}.csv'.format(csv_prefix, FEW_SHOT_NUM, 'control_signal' if IS_ADD_CONTROL_SIGNAL else '')


# set up model
logging.info('Loading {} to Device {}'.format(MODEL, DEVICE))
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = MT5ForConditionalGeneration.from_pretrained(MODEL).to(DEVICE)

# load dataset
logging.info('Loading dataset')
dataset = load_dataset("samsum")

# randomly sample 1000 dialogues from the dataset
df = dataset['train'].to_pandas()
df_test = dataset['test'].to_pandas()
# if sample_num is -1, then use all the training data
if SAMPLE_NUM == -1:
    SAMPLE_NUM = len(df)
train_data = df.sample(SAMPLE_NUM)
train_data = pd.concat([df_test, train_data], axis=0)
test_ids = df_test.id.tolist()
# log the index of the sampled dialogues
logging.info('Sampled dialogues: {}'.format(train_data.id.tolist()))

if IS_ADD_CONTROL_SIGNAL:
    logging.info('Extracting top 5 tf-idf keywords from the training dataset')
    # Initialize a TfidfVectorizer with the desired configuration
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=1, stop_words='english')

    if IS_UPPER_BOND:
        # Fit the vectorizer on the text data
        vectorizer.fit(train_data['summary'])

        # Extract the tf-idf matrix
        tfidf_matrix = vectorizer.transform(train_data['summary'])
    else:
        # Fit the vectorizer on the text data
        vectorizer.fit(train_data['dialogue'])

        # Extract the tf-idf matrix
        tfidf_matrix = vectorizer.transform(train_data['dialogue'])

    # Extract the feature names (i.e., the words) from the vectorizer
    try:
        feature_names = vectorizer.get_feature_names()
    except AttributeError:  # newer sklearn uses another function name,
        # ref: https://stackoverflow.com/questions/70215049/attributeerror-tfidfvectorizer-object-has-no-attribute
        # -get-feature-names-out
        feature_names = vectorizer.get_feature_names_out()

    # For each data record, extract the top 5 keywords based on their tf-idf scores
    keywords = []
    for i in range(len(train_data)):
        tfidf_scores = tfidf_matrix[i].toarray()[0]
        top_indices = tfidf_scores.argsort()[-NUMBER_OF_KEYWORDS:][::-1]
        top_keywords = [feature_names[idx] for idx in top_indices]
        keywords.append(top_keywords)

    # Add the keywords as a new column in the dataframe
    train_data['keywords'] = keywords


def generate_few_shot_example(train_df, few_shot_num, test_sample_ids=None):
    """
    Generate few-shot examples from the training dataset.
    :param test_sample_ids:
    :param train_df: pd.DataFrame, the training dataset
    :param few_shot_num: int, the number of few-shot examples
    :return: list of pd.DataFrame, each DataFrame contain the few-shot examples
    """
    # generate few-shot examples
    samples = []
    # remove ids in test_sample_ids from train_df
    if test_sample_ids:
        candidates = train_df[~train_df.id.isin(test_sample_ids)]
    else:
        candidates = train_df
    for index, _ in train_df.iterrows():
        s = candidates.drop(index).sample(few_shot_num)
        samples.append(s)
    # log the index of the sampled few-shot examples
    logging.info('Sampled few-shot examples: {}'.format([s.id.tolist() for s in samples]))
    return samples


few_shot_samples = generate_few_shot_example(train_data, FEW_SHOT_NUM, test_sample_ids=test_ids)

# prepare the prompt and append the prompt to train_data DataFrame
counter = 0
prompts = []


def formulate_record_to_prompt_text(dialogue, summary: str = None):
    prompt_text = 'Summarize the conversation:\n'
    dialogue = dialogue.strip().replace("\r", "")
    prompt_text += dialogue + '\n'
    prompt_text += 'Summary: '
    if summary:
        summary = summary.strip().replace("\n", "").replace("\r", "")
        prompt_text += summary + '</s>'
    return prompt_text


for _, row in train_data.iterrows():
    train_sample = few_shot_samples[counter]
    train_str = ""
    for _, sample in train_sample.iterrows():
        train_str += formulate_record_to_prompt_text(sample['dialogue'], sample['summary'])
        train_str += '\n'
        train_str += '\n'

    query_prompt = formulate_record_to_prompt_text(row['dialogue'])
    prompt = train_str + query_prompt
    prompts.append(prompt)

    counter += 1

train_data['processed_dialogue'] = prompts

# export train_data to csv file
train_data.to_csv('{}_train_data.csv'.format(csv_prefix))

# make inference without SAP
responses = []
spans_to_fill = []

prog_bar = tqdm(total=len(train_data))


def prompt_mT5(mt5_model, mt5_tokenizer, prompt_text: str, keywords_in_response: list = None):
    """
    Makes inference using mT5 model given the prompt text and keywords to include in the response.
    :param mt5_model:
    :param mt5_tokenizer:
    :param prompt_text: str
    :param keywords_in_response: list
    :return: the response text
    """
    # check if the keywords are provided
    if keywords_in_response is not None:
        # join each keyword with strings '<extra_id_i>', where i is incrementing from 0
        span_to_fill = '<extra_id_0> '  # empty space is needed
        mask_id = 1
        for keyword in keywords_in_response:
            span_to_fill += keyword + ' <extra_id_{}> '.format(mask_id)
            mask_id += 1
        spans_to_fill.append(span_to_fill.strip())
    else:
        span_to_fill = '<extra_id_0>'
    # tokenize dialogue data
    if IS_LOG_PROMPTS:
        logging.info('===Prompt===\n{}\n========='.format(prompt_text + span_to_fill))
    X = mt5_tokenizer.encode(prompt_text + span_to_fill, return_tensors="pt").to(DEVICE)
    # Summarize
    y_ids = mt5_model.generate(X, max_length=50, do_sample=False, eos_token_id=2, early_stopping=True, num_beams=5)
    y = tokenizer.decode(y_ids[0], skip_special_tokens=True)
    return y


for _, row in train_data.iterrows():
    prog_bar.update(1)
    response = prompt_mT5(model, tokenizer, row['processed_dialogue'],
                          row['keywords'] if IS_ADD_CONTROL_SIGNAL else None)

    responses.append(response)

# parse the responses
if 'keywords' in train_data.columns:
    # append the train data with two additional columns: "spans_to_fill" and "responses"
    train_data['spans_to_fill'] = spans_to_fill
    train_data['responses'] = responses
    # export the train data to csv file
    train_data.to_csv(csv_file_name, index=False)
    # exit the program
    sys.exit(0)
else:
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
