"""
Generate the pairs of demonstrations and test dataset and save the pairs in dictionary to a json file storing ids and
a pickle file.
These files ensure the models are tested with the same demonstrations.
"""

import nltk
from numpy import any
import numpy as np
import pickle
import json

import tqdm
from datasets import load_dataset

# Hyperparameters
RUN_NUM = 5
K = [1, 2, 3, 5]
IS_NUMERICAL_KEYWORDS = True
DATASET = 'samsum'  # some options: 'knkarthick/dialogsum'

# load the dataset
print("load the dataset")
train_dataset = load_dataset(DATASET, split='train')
test_dataset = load_dataset(DATASET, split='test')


def generate_for_k(k: int, is_numerical_keywords: bool = False):
    """
    :param is_numerical_keywords: bool, whether to use numerical keywords only, False means using tf-idf keywords
    :param k: int, the number of demonstrations
    """
    results_id = {}
    results = {}
    numerical_demonstrations = {
            'id': [],
            'dialogue': [],
            'summary': [],
            }

    if is_numerical_keywords:
        # get all training data from train_dataset
        for train_sample in tqdm.tqdm(train_dataset, total=len(train_dataset)):
            # check if train_sample's summary contains numerical information
            demo_keywords = [word for word in nltk.word_tokenize(train_sample['summary'])
                             if word.isdigit()]
            if len(demo_keywords) > 0:
                numerical_demonstrations['id'].append(train_sample['id'])
                numerical_demonstrations['dialogue'].append(train_sample['dialogue'])
                numerical_demonstrations['summary'].append(train_sample['summary'])
        # log the number of numerical demonstrations
        print("number of numerical demonstrations: {}".format(len(numerical_demonstrations['id'])))

    for run_id in range(RUN_NUM):
        print(f'Generate demonstrations for Run {run_id} in k={k}')
        results_id[run_id] = {}
        results[run_id] = {}

        # iterate over test samples to generate k demonstrations
        for test_sample in tqdm.tqdm(test_dataset, total=len(test_dataset)):
            # check if summary of test_sample has any numerical information
            if is_numerical_keywords:
                if not any([char.isdigit() for char in test_sample['summary']]):
                    continue
            test_id = test_sample['id']
            results_id[run_id][test_id] = {}
            # generate k demonstrations
            if is_numerical_keywords:
                demonstrations = {
                        'id': [],
                        'dialogue': [],
                        'summary': [],
                        }
                # random select k numerical demonstrations
                random_indices = np.random.choice(len(numerical_demonstrations['id']), k, replace=False)
                for i in range(k):
                    demonstrations['id'].append(numerical_demonstrations['id'][random_indices[i]])
                    demonstrations['dialogue'].append(numerical_demonstrations['dialogue'][random_indices[i]])
                    demonstrations['summary'].append(numerical_demonstrations['summary'][random_indices[i]])
            else:
                demonstrations = train_dataset.shuffle().select(range(k))[
                                 :]  # a dict with 3 keys: id, dialogue, summary
            # save the demonstrations information
            results_id[run_id][test_id] = demonstrations['id']
            results[run_id][test_id] = (test_sample, demonstrations)
    # export results_id to json
    dataset_name = DATASET.replace('/', '_')  # replace  / in DATASET with _
    with open(f'demonstration_{dataset_name}_k{k}_{is_numerical_keywords}.json', 'w') as f:
        json.dump(results_id, f)
    # export results to pickle
    with open(f'demonstration_{dataset_name}_k{k}_{is_numerical_keywords}.pickle', 'wb') as f:
        pickle.dump(results, f)
    # load the results from pickle for validation purpose
    with open(f'demonstration_{dataset_name}_k{k}_{is_numerical_keywords}.pickle', 'rb') as f:
        pickle.load(f)
        print('Validation succeeds')


if isinstance(K, int):
    generate_for_k(K, IS_NUMERICAL_KEYWORDS)
elif isinstance(K, list):
    for k_ in K:
        generate_for_k(k_, IS_NUMERICAL_KEYWORDS)
else:
    raise ValueError(f'K should be int or list, but got {type(K)}')
