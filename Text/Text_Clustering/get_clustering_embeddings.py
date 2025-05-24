import argparse
import os
import json
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel
from sklearn.datasets import fetch_20newsgroups

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Embedding Generator")
    parser.add_argument("--dataset", type=str, required=True, choices=[
        "biorxiv-p2p", "biorxiv-s2s", "twentynewsgroups"
    ], help="Choose dataset: biorxiv-p2p, biorxiv-s2s, or twentynewsgroups")
    parser.add_argument("--split", type=str, required=True, choices=[
        "train", "test", "val", "all"
    ], help="Choose split: train, test, val, all (twentynewsgroups only uses all)")
    return parser.parse_args()

def process_biorxiv(file_path, embed_path, instruction, batch_size):
    string_to_number = {}
    with open(file_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            label = obj['labels']
            if label not in string_to_number:
                string_to_number[label] = len(string_to_number)

    texts, labels = [], []
    text_embeddings_buffer, text_label_buffer = [], []

    with open(file_path, "r") as f:
        for line in tqdm(f, desc="Processing"):
            obj = json.loads(line)
            texts.append(obj['sentences'])
            labels.append(string_to_number[obj['labels']])
            if len(texts) == batch_size:
                emb = model.encode(texts, instruction=instruction)
                emb = F.normalize(emb, p=2, dim=1)
                text_embeddings_buffer.append(emb.cpu().numpy())
                text_label_buffer.extend(labels)
                texts, labels = [], []

        if texts:
            emb = model.encode(texts, instruction=instruction)
            emb = F.normalize(emb, p=2, dim=1)
            text_embeddings_buffer.append(emb.cpu().numpy())
            text_label_buffer.extend(labels)

    np.savez(embed_path, data=np.vstack(text_embeddings_buffer), label=np.array(text_label_buffer))

def process_twentynewsgroups(embed_path, instruction, batch_size, split):
    data = fetch_20newsgroups(subset="all" if split == "all" else split, remove=('headers', 'footers', 'quotes'))
    texts, labels = data.data, data.target

    text_embeddings_buffer, text_label_buffer = [], []
    batch_texts, batch_labels = [], []

    for t, l in tqdm(zip(texts, labels), desc="Processing", total=len(texts)):
        batch_texts.append(t)
        batch_labels.append(l)
        if len(batch_texts) == batch_size:
            emb = model.encode(batch_texts, instruction=instruction)
            emb = F.normalize(emb, p=2, dim=1)
            text_embeddings_buffer.append(emb.cpu().numpy())
            text_label_buffer.extend(batch_labels)
            batch_texts, batch_labels = [], []
    if batch_texts:
        emb = model.encode(batch_texts, instruction=instruction)
        emb = F.normalize(emb, p=2, dim=1)
        text_embeddings_buffer.append(emb.cpu().numpy())
        text_label_buffer.extend(batch_labels)

    np.savez(f"{embed_path}_{split}.npz", data=np.vstack(text_embeddings_buffer), label=np.array(text_label_buffer))

if __name__ == "__main__":
    args = parse_args()

    model_name = "nvidia/NV-Embed-v2"
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto")

    DATASET_CONFIG = {
        "biorxiv-p2p": {
            "dir": "../datasets/biorxiv-clustering-p2p/",
            "instruction": "Generate a representation that captures the core scientific topic and main subject of the following title and abstract for the purpose of clustering similar research together. Focus on themes, methods, and biological concepts. \n Title and abstract: ",
            "batch_size": 8,
            "file_template": "{split}.jsonl",
            "output": "./biorxiv-clustering-p2p/{split}_gpt.npz",
            "processor": process_biorxiv
        },
        "biorxiv-s2s": {
            "dir": "../datasets/biorxiv-clustering-s2s/",
            "instruction": "Given a title of an essay related to biology, Please describe the research field this essay may belong to. \n Title: ",
            "batch_size": 2,
            "file_template": "{split}.jsonl",
            "output": "./biorxiv-clustering-s2s/{split}_gpt.npz",
            "processor": process_biorxiv
        },
        "twentynewsgroups": {
            "dir": "../datasets/twentynewsgroups-clustering/",
            "instruction": "Identify the topic or theme of the given news articles. \n News article: ",
            "batch_size": 2,
            "file_template": "{split}.jsonl",
            "output": "./twentynewsgroups/my_prompt",
            "processor": process_twentynewsgroups
        }
    }

    dataset_cfg = DATASET_CONFIG[args.dataset]
    
    if args.dataset == "twentynewsgroups" and args.split not in ["all", "train", "test"]:
        raise ValueError("twentynewsgroups only supports all, train and test splits")

    if args.dataset == "biorxiv-s2s" or args.dataset == 'biorxiv-p2p' and args.split not in ["test"]:
        raise ValueError("biorxiv-datasets only supports test splits")

    input_file = os.path.join(dataset_cfg["dir"], dataset_cfg["file_template"].format(split=args.split))
    output_path = dataset_cfg["output"].format(split=args.split)

    if args.dataset.startswith("biorxiv"):
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file does not exist: {input_file}")
        dataset_cfg["processor"](input_file, output_path, dataset_cfg["instruction"], dataset_cfg["batch_size"])
    elif args.dataset == "twentynewsgroups":
        dataset_cfg["processor"](output_path, dataset_cfg["instruction"], dataset_cfg["batch_size"], args.split)
    else:
        raise ValueError("Unknown dataset type")

    print(f"Embeddings saved to {output_path if args.dataset != 'twentynewsgroups' else output_path + '_' + args.split + '.npz'}")
