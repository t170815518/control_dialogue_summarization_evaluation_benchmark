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


# hyper-parameters
MODEL = 'google/mt5-xl'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_NUM = 1000
FEW_SHOT_NUM = 1
IS_ADD_CONTROL_SIGNAL = True


# remove any special character in MODEL as the prefix of csv file
csv_prefix = MODEL.replace('/', '_')
csv_file_name = '{}_{}shot_{}.csv'.format(csv_prefix, FEW_SHOT_NUM, 'control_signal' if IS_ADD_CONTROL_SIGNAL else '')


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


if IS_ADD_CONTROL_SIGNAL:
    logging.info('Extracting top 5 tf-idf keywords from the training dataset')
    # Initialize a TfidfVectorizer with the desired configuration
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')

    # Fit the vectorizer on the text data
    vectorizer.fit(train_data['dialogue'])

    # Extract the tf-idf matrix
    tfidf_matrix = vectorizer.transform(train_data['dialogue'])

    # Extract the feature names (i.e., the words) from the vectorizer
    feature_names = vectorizer.get_feature_names()

    # For each data record, extract the top 5 keywords based on their tf-idf scores
    keywords = []
    for i in range(len(train_data)):
        tfidf_scores = tfidf_matrix[i].toarray()[0]
        top_indices = tfidf_scores.argsort()[-5:][::-1]
        top_keywords = [feature_names[idx] for idx in top_indices]
        keywords.append(top_keywords)

    # Add the keywords as a new column in the dataframe
    train_data['keywords'] = keywords


def generate_few_shot_example(train_df, few_shot_num):
    """
    Generate few-shot examples from the training dataset.
    :param train_df: pd.DataFrame, the training dataset
    :param few_shot_num: int, the number of few-shot examples
    :return: list of pd.DataFrame, each DataFrame contain the few-shot examples
    """
    # generate few-shot examples
    samples = []
    for index, _ in train_df.iterrows():
        candidates = df.drop(index)
        s = candidates.sample(few_shot_num)
        samples.append(s)
    # log the index of the sampled few-shot examples
    logging.info('Sampled few-shot examples: {}'.format([s.id.tolist() for s in samples]))
    return samples


few_shot_samples = generate_few_shot_example(train_data, FEW_SHOT_NUM)


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

    prompt = "summarize the conversation.\n" + train_str + "\n" + 'Conversation: '\
             + row['dialogue'].strip().replace("\n", " ").replace("\r", " ") \
             + '\nsummary: '
    prompts.append(prompt)

    counter += 1

train_data['processed_dialogue'] = prompts


# make inference without SAP
responses = []
spans_to_fill = []

prog_bar = tqdm(total=len(train_data))
for _, row in train_data.iterrows():
    prog_bar.update(1)
    prompt = row['processed_dialogue']
    # check if the row has "keywords" column
    if 'keywords' in row:
        # join each keyword with strings '<extra_id_i>', where i is incrementing from 0
        keywords = row['keywords']
        span_to_fill = '<extra_id_0> '  # empty space is needed
        mask_id = 1
        for keyword in keywords:
            span_to_fill += keyword + ' <extra_id_{}>'.format(mask_id)
            mask_id += 1
        spans_to_fill.append(span_to_fill)
    else:
        span_to_fill = '<extra_id_0>'
    # tokenize dialogue data
    X = tokenizer.encode(prompt + span_to_fill, return_tensors="pt").to(DEVICE)
    # Summarize
    y_ids = model.generate(X,
                           num_beams=4,
                           no_repeat_ngram_size=2,
                           early_stopping=True)
    y = tokenizer.decode(y_ids[0], skip_special_tokens=True)

    responses.append(y)


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
