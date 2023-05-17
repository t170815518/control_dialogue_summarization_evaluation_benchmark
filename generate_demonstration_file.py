"""
Generate the pairs of demonstrations and test dataset and save the pairs in dictionary to a json file.
The pickle file ensure the models are tested with the same demonstrations.
"""


import json

import tqdm
from datasets import load_dataset


RUN_NUM = 5
K = 5
DATASET = 'samsum'


# load the dataset
train_dataset = load_dataset(DATASET, split='train')
test_dataset = load_dataset(DATASET, split='test')

results = {}
for run_id in range(RUN_NUM):
    print(f'Generate demonstrations for Run {run_id} in k={K}')
    results[run_id] = {}

    # iterate over test samples to generate k demonstrations
    for test_sample in tqdm.tqdm(test_dataset, total=len(test_dataset)):
        test_id = test_sample['id']
        results[run_id][test_id] = {}
        # generate k demonstrations
        demonstrations_id = train_dataset.shuffle().select(range(K))[:]['id']
        results[run_id][test_id] = demonstrations_id

# export results to json
with open(f'demonstration_{DATASET}_k{K}.json', 'w') as f:
    json.dump(results, f)
