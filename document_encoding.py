"""
Gets document embedding using SBERT pre-trained model, and stores embedding in hdf5 format
@Author: Tang Yuting
@Update Date: 15 Jan, 2023

Notes:
    - cnn_dailymail dataset uses version 3.0
"""


import h5py
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


# change the arguments below
PRETRAINED_PATH = 'sentence-transformers/all-MiniLM-L6-v2'
DATASET_NAME = 'xsum'  # used to load huggingface dataset
SUB_DATASET_NAME = 'train'
BATCH_SIZE = 500  # inference is done in batches
assert PRETRAINED_PATH != ''
assert DATASET_NAME != ''
assert BATCH_SIZE > 0


# load dataset
if DATASET_NAME == 'cnn_dailymail':
    dataset = load_dataset("cnn_dailymail", '3.0.0')
    text_column = 'article'  # the document column name varies in different datasets
else:
    dataset = load_dataset(DATASET_NAME)
    text_column = 'document'


# load pre-trained transformer
model = SentenceTransformer(PRETRAINED_PATH)


# save data in hd5f format
h5f = h5py.File('{}.h5'.format(DATASET_NAME), 'w')
# get the embedding of documents in batches
prog_bar = tqdm(total=len(dataset[SUB_DATASET_NAME]))
id2embedding = {}
for start_id in range(0, len(dataset[SUB_DATASET_NAME]), BATCH_SIZE):
    end_id = min(len(dataset[SUB_DATASET_NAME]), start_id + BATCH_SIZE)
    documents = dataset[SUB_DATASET_NAME][start_id:end_id]
    document_texts = documents[text_column]
    document_ids = documents['id']
    # encode the documents in batches
    embeddings = model.encode(document_texts)  # output is tuple with length of BATCH_SIZE
    # save the embeddings to local path
    for index, embedding in zip(document_ids, embeddings):
        h5f.create_dataset(index, data=embedding)
    prog_bar.update(len(embeddings))


h5f.close()
