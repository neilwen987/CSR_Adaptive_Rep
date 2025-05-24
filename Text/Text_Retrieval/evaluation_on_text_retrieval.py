import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import os

def get_topk_corpus(query_embed, corpus_embeds, topk=10):
    query_embed = F.normalize(query_embed, dim=1)
    corpus_embeds = F.normalize(corpus_embeds, dim=1)
    similarities = torch.matmul(corpus_embeds, query_embed.T).squeeze()
    topk_indices = torch.topk(similarities, k=topk).indices.cpu().numpy()
    return topk_indices

def dcg(relevances, k=10):
    relevances = np.asfarray(relevances)[:k]
    if relevances.size:
        return np.sum((2**relevances - 1) / np.log2(np.arange(2, relevances.size + 2)))
    return 0.0

def ndcg(retrieved_ids, relevant_ids, k=10):
    relevances = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved_ids[:k]]
    ideal_relevances = sorted(relevances, reverse=True)
    return dcg(relevances, k) / dcg(ideal_relevances, k) if dcg(ideal_relevances, k) > 0 else 0.0

def load_jsonl_text_and_ids(jsonl_path, text_key="text", id_key="_id"):
    data = []
    data_id = []
    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj[text_key])
            data_id.append(obj[id_key])
    return data, data_id

def load_relevance_dict(qrels_path):
    relevance_dict = {}
    tsv_file = pd.read_csv(qrels_path, delimiter='\t')
    for _, row in tsv_file.iterrows():
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        relevance_dict.setdefault(qid, []).append(cid)
    return relevance_dict

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading embeddings...")
    corpus_embed_np = np.load(args.corpus_embed_path)["data"]
    queries_embed_np = np.load(args.queries_embed_path)["data"]
    corpus_embed = torch.tensor(corpus_embed_np, dtype=torch.float32).to(device)
    queries_embed = torch.tensor(queries_embed_np, dtype=torch.float32).to(device)

    print("Loading corpus and queries...")
    corpus, corpus_id = load_jsonl_text_and_ids(args.corpus_jsonl)
    queries, queries_id = load_jsonl_text_and_ids(args.queries_jsonl)

    print("Loading relevance scores...")
    relevance_dict = load_relevance_dict(args.qrels_path)

    print("Evaluating queries for NDCG@%d..." % args.k)
    ndcg_scores = []
    for i, query_embed in enumerate(tqdm(queries_embed)):
        qid = str(queries_id[i])
        if qid not in relevance_dict:
            continue
        relevant_corpus_ids = relevance_dict[qid]
        query_embed_gpu = query_embed.unsqueeze(0)
        retrieved_indices = get_topk_corpus(query_embed_gpu, corpus_embed, topk=args.k)
        retrieved_ids = [corpus_id[idx] for idx in retrieved_indices]
        score = ndcg(retrieved_ids, relevant_corpus_ids, k=args.k)
        ndcg_scores.append(score)

    average_ndcg = np.mean(ndcg_scores)
    print(f"Average NDCG@{args.k}: {average_ndcg:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NDCG@K for dense retrieval embeddings.")
    parser.add_argument("--corpus_embed_path", type=str, required=True, help="Path to corpus embedding npz file")
    parser.add_argument("--queries_embed_path", type=str, required=True, help="Path to queries embedding npz file")
    parser.add_argument("--corpus_jsonl", type=str, required=True, help="Path to corpus JSONL file")
    parser.add_argument("--queries_jsonl", type=str, required=True, help="Path to queries JSONL file")
    parser.add_argument("--qrels_path", type=str, required=True, help="Path to qrels TSV file")
    parser.add_argument("--k", type=int, default=10, help="Evaluate NDCG@K (default: 10)")
    args = parser.parse_args()
    main(args)
