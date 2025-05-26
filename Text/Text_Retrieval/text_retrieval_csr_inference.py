import sys 
sys.path.append("../") # adding root folder to the path
import os

import torch 
import torchvision
import timm
from torchvision import transforms
from torchvision.models import *
from torchvision import datasets
from tqdm import tqdm
from argparse import ArgumentParser
from utils import *
from model_zoo import *
from collections import OrderedDict

parser=ArgumentParser()

# model args
parser.add_argument('--corpus_emb_path', default='../retrieval/pretrained_emb/train_emb',
                    help='path to pre-trained training embeddings (default: imagenet)')
parser.add_argument('--queries_emb_path', default='../retrieval/pretrained_emb/val_emb',
                    help='path to pre-trained evaluation embeddings (default: imagenet)')
parser.add_argument('--retrieval_array_path', default='./retrieval',
                    help='path to save database and query arrays for retrieval', type=str)
parser.add_argument('--workers', type=int, default=16, help='num workers for dataloader')
parser.add_argument('--batch_size', default=256,type=int,help='batch size')
parser.add_argument('--embed_save_path', default='../retrieval/pretrained_emb', help='path to save database and query arrays for retrieval', type=str)
parser.add_argument('--embed_dim', default=4096 ,help='embedding_dim')

# CSR args
parser.add_argument('--topk', default=8, type=int,help='the number of topk')
parser.add_argument('--auxk', default=512, type=int,help='the number of auxk')
parser.add_argument('--auxk_coef', default=1/32, type=float,help='auxk_coef')
parser.add_argument('--dead_threshold', default=10, type=int,help='dead_threshold')
parser.add_argument('--hidden-size', default=None, type=int,help='the size of hidden layer')
parser.add_argument('--csr_ckpt', default=None,type=str,help = 'ckpt for CSR model')

args = parser.parse_args()


model = CSR(n_latents=args.hidden_size, topk=args.topk, auxk=args.auxk, dead_threshold=args.dead_threshold,
            normalize=False, n_inputs=args.embed_dim)

state = torch.load(args.csr_ckpt, map_location='cpu')['state_dict']
state_clean = OrderedDict()
for key in state.keys():
    if key.startswith('module.'):
        state_clean[key.replace("module.", "")] = state[key]
    else:
        state_clean[key] = state[key]

model.load_state_dict(state_clean)
model = model.cuda()
model.eval()


print("Inferencing Training Dataset")
generate_retrieval_data(model, args.corpus_emb_path,args.retrieval_array_path, args, "corpus")

print("Inferencing Evaluation Dataset")
generate_retrieval_data(model, args.queries_emb_path,args.retrieval_array_path, args, "queries")
