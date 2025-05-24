import argparse
import os
import torch.nn.functional as F
from transformers import AutoModel
import json
from tqdm import tqdm
import numpy as np

DATASET_CONFIG = {
    "banking77": {
        "instruction": "Instruct: Given a question, please describe the intent of this question. \n Question: ",
        "splits": ["train", "test"],
        "file_template": "../datasets/banking77/{split}.jsonl",
        "embed_template": "./Banking77/{split}.npz",
        "languages": None, 
    },
    "tweet_sentiment_extraction": {
        "instruction": "Instruct: Given a sentence, please describe the sentiment behind it. Whether is it positive, neutral or negative? \n Sentence: ",
        "splits": ["train", "test"],
        "file_template": "../datasets/tweet_sentiment_extraction/{split}.jsonl",
        "embed_template": "./tweet_sentiment_extraction/{split}.npz",
        "languages": None, 
    },
    "mtop_intent": {
        "instruction": "Instruct: Given a question, please describe the intent of this question. \n Question: ",
        "splits": ["train", "validation", "test"],
        "languages": ['de', 'en', 'es', 'fr', 'hi', 'th'],
        "file_template": "../datasets/mtop_intent/{language}/{split}.jsonl",
        "embed_template": "./mtop_intent/{language}/{split}.npz",
    }
}

def process_file(file_path, embed_path, model, instruction, batch_size=128):
    text_label_buffer = []
    text_embeddings_buffer = []
    texts, labels = [], []
    if not os.path.exists(os.path.dirname(embed_path)):
        os.makedirs(os.path.dirname(embed_path), exist_ok=True)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Processing {os.path.basename(file_path)}"):
            obj = json.loads(line)
            texts.append(obj['text'])
            labels.append(obj['label'])
            if len(texts) == batch_size:
                embeddings = model.encode(texts, instruction=instruction)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                text_embeddings_buffer.append(embeddings.cpu().numpy())
                text_label_buffer.extend(labels)
                texts, labels = [], []

        if texts:
            embeddings = model.encode(texts, instruction=instruction)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            text_embeddings_buffer.append(embeddings.cpu().numpy())
            text_label_buffer.extend(labels)

    embedding_array = np.vstack(text_embeddings_buffer)
    labels_array = np.array(text_label_buffer)
    np.savez(embed_path, data=embedding_array, label=labels_array)

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for various NLP datasets.")
    parser.add_argument('--dataset', required=True, choices=DATASET_CONFIG.keys(), help="Dataset to process")
    parser.add_argument('--language', default=None, help="Language (required for mtop_intent)")
    parser.add_argument('--split', default=None, help="Data split: train/test/validation (default: all splits)")

    args = parser.parse_args()
    config = DATASET_CONFIG[args.dataset]

    model_name = "nvidia/NV-Embed-v2"
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto")

    if config["languages"]: 
        languages = [args.language] if args.language else config["languages"]
        splits = [args.split] if args.split else config["splits"]
        for language in languages:
            for split in splits:
                file_path = config["file_template"].format(language=language, split=split)
                embed_path = config["embed_template"].format(language=language, split=split)
                if not os.path.exists(file_path):
                    print(f"Warning: File not found {file_path}, skipping.")
                    continue
                print(f"Processing {args.dataset} | language={language}, split={split}")
                process_file(file_path, embed_path, model, config["instruction"])
    else:
        splits = [args.split] if args.split else config["splits"]
        for split in splits:
            file_path = config["file_template"].format(split=split)
            embed_path = config["embed_template"].format(split=split)
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}, skipping.")
                continue
            print(f"Processing {args.dataset} | split={split}")
            process_file(file_path, embed_path, model, config["instruction"])

if __name__ == "__main__":
    main()
