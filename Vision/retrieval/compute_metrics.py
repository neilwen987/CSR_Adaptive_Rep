import pandas as pd
import numpy as np
import time
import os
from argparse import ArgumentParser

parser=ArgumentParser()
parser.add_argument('--topk', default=8, type=int,help='the number of topk')
args = parser.parse_args()
topk = args.topk
root = f"./CSR_topk_{topk}/"
dataset = 'V1'
index_type = 'exactl2' # ['exactl2', 'hnsw_8', 'hnsw_32']
EVAL_CONFIG = 'vanilla' # ['vanilla', 'reranking', 'funnel']
k = 2048 # shortlist length, default set to max supported by FAISS
def compute_mAP_recall_at_k(val_classes, db_classes, neighbors, k):
    """
    Computes the MAP@k (default value of k=R) on neighbors with val set by seeing if nearest neighbor
    is in the same class as the class of the val code. Let m be size of val set, and n in train.

      val:          (m x d) All the truncated vector representations of images in val set
      val_classes:  (m x 1) class index values for each vector in the val set
      db_classes:   (n x 1) class index values for each vector in the train set
      neighbors:    (k x m) indices in train set of top k neighbors for each vector in val set
    """

    topk = []
    for i in range(val_classes.shape[0]):  # Compute precision for each vector's list of k-nn
        target = val_classes[i]
        indices = neighbors[i, :][:k]  # k neighbor list for ith val vector
        labels = db_classes[indices]
        matches = (labels == target)

        # topk
        hits = np.sum(matches)
        if hits > 0:
            topk.append(1)
        else:
            topk.append(0)


    return np.mean(topk)


# Load database and query set for nested models

# Database: 1.2M x 1 for imagenet1k
db_labels = np.load(root + f"V1 train topk{topk}-y.npy")

# Query set: 50K x 1 for imagenet1k
query_labels = np.load(root + f"V1 eval topk{topk}-y.npy")


start = time.time()
# Load database and query set for fixed feature models
# Load neighbors array and compute metrics

neighbors_path = root + "neighbors/" +index_type + "_" \
                 + f"{k}shortlist_" + dataset + ".csv"

neighbors = pd.read_csv(neighbors_path, header=None).to_numpy()

top1 = db_labels[neighbors[:, 0]]
print("Top1= ", np.sum(top1 == query_labels) / query_labels.shape[0])

end = time.time()
print("Eval time for SAE = %0.3f sec\n" % (end - start))
