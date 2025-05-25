import numpy as np
import faiss
import time
import pandas as pd
from argparse import ArgumentParser
from os import path, makedirs

parser=ArgumentParser()
parser.add_argument('--topk', default=8, type=int,help='the number of topk')
args = parser.parse_args()

topk = args.topk
root = f'./CSR_topk_{topk}/'
dataset = 'V1'
index_type = 'exactl2'
k = 2048 # shortlist length, default set to max supported by FAISS

db_csv = f'{dataset} train topk{topk}'+ '-X.npy'
query_csv = f'{dataset} eval topk{topk}'+ '-X.npy'

ngpus = faiss.get_num_gpus()
print("number of GPUs:", ngpus)

start = time.time()
database = np.load(root+db_csv)
queryset = np.load(root+query_csv)
print("CSV file load time= %0.3f sec" % (time.time() - start))

if index_type == 'exactl2':
    use_gpu = 1 # GPU inference for exact search
else:
    use_gpu = 0 # GPU inference for HNSW is currently not supported by FAISS


if not path.isdir(root + 'index_files/'):
    makedirs(root + 'index_files/')

index_file = root + 'index_files/' + dataset + '_' + index_type + '.index'
# Load or build index
if path.exists(index_file):
    print("Loading index file: " + index_file)
    cpu_index = faiss.read_index(index_file)

else:
    print("Generating index file: " + index_file)

    xb = np.ascontiguousarray(np.load(root + db_csv), dtype=np.float32)
    faiss.normalize_L2(xb)
    d = xb.shape[1]  # dimension
    nb = xb.shape[0]  # database size
    print("database: ", xb.shape)

    start = time.time()
    if index_type == 'exactl2':
        print("Building Exact L2 Index")
        cpu_index = faiss.IndexFlatL2(d)  # build the index

    cpu_index.add(xb)  # add vectors to the index
    faiss.write_index(cpu_index, index_file)
    print("GPU Index build time= %0.3f sec" % (time.time() - start))


index = cpu_index
    # Load the queries
xq = np.ascontiguousarray(np.load(root + query_csv), dtype=np.float32)
faiss.normalize_L2(xq)
nq = xq.shape[0]
print("queries: ", xq.shape)

start = time.time()
D, I = index.search(xq, k)
end = time.time() - start
print("GPU %d-NN search time= %f sec" % (k, end))

start = time.time()
if not path.isdir(root + "neighbors/"):
    makedirs(root + "neighbors/")
nn_dir = root + "neighbors/" + index_type + "_" + str(
    k) + "shortlist_" + dataset + ".csv"
pd.DataFrame(I).to_csv(nn_dir, header=None, index=None)
end = time.time() - start
print("NN file write time= %0.3f sec\n" % (end))
