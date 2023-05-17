"""
Generate the pairs of demonstrations and test dataset and save the pairs in dictionary to a json file storing ids and
a pickle file.
These files ensure the models are tested with the same demonstrations.
"""


import pickle
import json

import tqdm
from datasets import load_dataset


# Hyperparameters
RUN_NUM = 5
K = [1, 2, 3, 5]
DATASET = 'samsum'


# load the dataset
print("load the dataset")
train_dataset = load_dataset(DATASET, split='train')
test_dataset = load_dataset(DATASET, split='test')


def generate_for_k(k: int):
    """
    :param k: int, the number of demonstrations
    """
    results_id = {}
    results = {}
    for run_id in range(RUN_NUM):
        print(f'Generate demonstrations for Run {run_id} in k={k}')
        results_id[run_id] = {}
        results[run_id] = {}

        # iterate over test samples to generate k demonstrations
        for test_sample in tqdm.tqdm(test_dataset, total=len(test_dataset)):
            test_id = test_sample['id']
            results_id[run_id][test_id] = {}
            # generate k demonstrations
            demonstrations = train_dataset.shuffle().select(range(k))[:]
            # save the demonstrations information
            results_id[run_id][test_id] = demonstrations['id']
            results[run_id][test_id] = (test_sample, demonstrations)
    # export results_id to json
    with open(f'demonstration_{DATASET}_k{k}.json', 'w') as f:
        json.dump(results_id, f)
    # export results to pickle
    with open(f'demonstration_{DATASET}_k{k}.pickle', 'wb') as f:
        pickle.dump(results, f)
    # load the results from pickle for validation purpose
    with open(f'demonstration_{DATASET}_k{k}.pickle', 'rb') as f:
        pickle.load(f)
        print('Validation succeeds')


if isinstance(K, int):
    generate_for_k(K)
elif isinstance(K, list):
    for k_ in K:
        generate_for_k(k_)
else:
    raise ValueError(f'K should be int or list, but got {type(K)}')
