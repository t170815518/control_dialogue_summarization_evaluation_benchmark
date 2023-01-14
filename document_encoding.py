"""
Gets document embedding using SBERT pre-trained model.
@Author: Tang Yuting
@Update Date: 14 Jan, 2023

fixme: some attributes need to change according to different dataset configuration
"""


import os

import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


DATASET_NAME = 'cnn_dailymail'  # used to load huggingface dataset
BATCH_SIZE = 500  # inference is done in batches
assert DATASET_NAME != ''
assert BATCH_SIZE > 0


output_dir = 'output_' + DATASET_NAME  # by default, the embeddings are saved in local folder with the same 'output_'
# + dataset
if not os.path.exists(os.path.join('..', output_dir)):  # create the folder if not exists
    os.mkdir(os.path.join('..', output_dir))


# load dataset
if DATASET_NAME == 'cnn_dailymail':
    dataset = load_dataset("cnn_dailymail", '3.0.0')
else:
    dataset = load_dataset(DATASET_NAME)


# load pre-trained transformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# get the embedding of documents in batches
prog_bar = tqdm(total=len(dataset['train']))
for start_id in range(0, len(dataset['train']), BATCH_SIZE):
    end_id = min(len(dataset['train']), start_id + BATCH_SIZE)
    documents = dataset['train'][start_id:end_id]
    document_texts = documents['article']
    document_ids = documents['id']
    # encode the documents in batches
    embeddings = model.encode(document_texts)   # output is tuple with length of BATCH_SIZE
    # save the embeddings to local path
    for index, embedding in zip(document_ids, embeddings):
        np.save(os.path.join('..', output_dir, '{}.npy'.format(index)), embedding)
    prog_bar.update(len(embeddings))
