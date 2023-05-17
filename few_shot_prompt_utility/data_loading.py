import logging
import json
import tqdm
from functools import lru_cache
from datasets import load_dataset


@lru_cache(maxsize=None)
def load_dialogue_pairs(demonstration_pair_path: str, dataset_name: str):
    """
    Load the dialogue pairs from the demonstration_pair_path and extract the dialogue records from the
    Huggingface dataset.
    :param demonstration_pair_path: str, the path to the demonstration pair file in json format
    :param dataset_name: str, the name of the HuggingFace dataset
    :return: dict, a dictionary with the key as the run id and the value as the list of dialogue pairs like
    (demonstration_dialogues, test_dialogue)
    """
    # read the demonstration file in json format
    with open(demonstration_pair_path, 'r') as f:
        demonstration_dict = json.load(f)

    logging.debug("load the dataset")
    train_dataset = load_dataset(dataset_name, split='train')
    test_dataset = load_dataset(dataset_name, split='test')

    # extract the dialogue records from the HuggingFace dataset
    run_id2dialogue_pairs = {}
    for run_id, pairs in demonstration_dict.items():
        logging.info(f'Extract dialogue pairs for run {run_id}')
        run_id2dialogue_pairs[run_id] = []
        for test_id, demonstrations_id in tqdm.tqdm(pairs.items()):
            # extract test_id from test dataset and demonstrations_id from train dataset
            test_sample = test_dataset.filter(lambda x_: x_['id'] == test_id)[0]
            demonstrations_samples = train_dataset.filter(lambda x_: x_['id'] in demonstrations_id)[:]
            test_sample = test_dataset.select([test_id])[0]
            demonstrations_samples = train_dataset.select(demonstrations_id)[:]
            run_id2dialogue_pairs[run_id].append((demonstrations_samples, test_sample))
        # log the number of dialogue pairs for each run id
        logging.debug(f'Run {run_id} has {len(run_id2dialogue_pairs[run_id])} dialogue pairs')
    return run_id2dialogue_pairs


if __name__ == '__main__':
    # load the dataset and extract samples according to the indexes in the demonstration file
    DATASET = 'samsum'
    DEMONSTRATION_PAIR_PATH = 'demonstration_pairs/demonstration_samsum_k1.json'
    x = load_dialogue_pairs(DEMONSTRATION_PAIR_PATH, DATASET)
    print(x)
