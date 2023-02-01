"""
Gets related documents by computing cosine similarity between two documents' embeddings and exports the results to
hdf5 format
@Author: Tang Yuting
@Update Date: 1 Feb, 2023
"""


from collections import defaultdict
import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# hyper-parameters
THRESHOLD = 0.7
BATCH_SIZE = 500


# todo: change the arguments below
xsum_h5f = h5py.File('xsum.h5', 'r')
cnn_h5f = h5py.File('cnn_dailymail.h5', 'r')


# get the embeddings of documents
xsum_mat = []
xsum_keys = []
for key in tqdm(xsum_h5f):
    xsum_mat.append(xsum_h5f[key][:])
    xsum_keys.append(key)
xsum_mat = np.stack(xsum_mat)
cnn_mat = []
cnn_keys = []
for key in tqdm(cnn_h5f):
    cnn_mat.append(cnn_h5f[key][:])
    cnn_keys.append(key)
cnn_mat = np.stack(cnn_mat)


# compute the similar documents id in batches
xsum2cnn_related = defaultdict(list)
for start_id in tqdm(range(0, len(xsum_mat), BATCH_SIZE)):
    end_id = min(start_id + BATCH_SIZE, len(xsum_mat))
    similarity_mat = cosine_similarity(X=xsum_mat[start_id:end_id, :], Y=cnn_mat)
    xsum_indexes = xsum_keys[start_id:end_id]
    x_indexes, y_indexes = np.where(similarity_mat >= THRESHOLD)
    for x, y in zip(x_indexes, y_indexes):
        xsum2cnn_related[xsum_indexes[x]].append(cnn_keys[y])


# export the similar documents in hdf5 format
h5f = h5py.File('xsum2cnn_related.h5', 'w')
for key, value in xsum2cnn_related.items():
    h5f.create_dataset(key, data=np.array(value))
h5f.close()
