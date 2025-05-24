import argparse
import torch.nn.functional as F
from transformers import AutoModel
import json
from tqdm import tqdm
import numpy as np
import os


def get_model():
    model_name = "nvidia/NV-Embed-v2"
    return AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto")

def default_processor(model, input_file, output_file, instruction, batch_size):
    text_embeddings_buffer = []
    texts = []

    with open(input_file, "r") as f:
        for line in tqdm(f, desc=f"Processing {input_file}"):
            obj = json.loads(line)
            texts.append(obj['text'])

            if len(texts) == batch_size:
                embeddings = model.encode(texts, instruction=instruction)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                text_embeddings_buffer.append(embeddings.cpu().numpy())
                texts = []

        if texts:
            embeddings = model.encode(texts, instruction=instruction)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            text_embeddings_buffer.append(embeddings.cpu().numpy())

    embedding_array = np.vstack(text_embeddings_buffer)
    np.savez(output_file, data=embedding_array)
    print(f"Saved embeddings to {output_file}")

def process_biorxiv(model, input_file, output_file, instruction, batch_size):
    default_processor(model, input_file, output_file, instruction, batch_size)

def process_twentynewsgroups(model, input_file, output_file, instruction, batch_size):
    default_processor(model, input_file, output_file, instruction, batch_size)

def process_fiqa(model, input_file, output_file, instruction, batch_size):
    text_embeddings_buffer = []
    texts = []

    with open(input_file, "r") as f:
        for line in tqdm(f, desc=f"Processing {input_file}"):
            obj = json.loads(line)
            texts.append(obj['text'])

            if len(texts) == batch_size:
                embeddings = model.encode(texts, instruction=instruction)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                text_embeddings_buffer.append(embeddings.cpu().numpy())
                texts = []

        if texts:
            embeddings = model.encode(texts, instruction=instruction)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            text_embeddings_buffer.append(embeddings.cpu().numpy())

    embedding_array = np.vstack(text_embeddings_buffer)
    np.savez(output_file, data=embedding_array)
    print(f"Saved embeddings to {output_file}")

DATASET_CONFIG = {
    "nfcorpus": {
        "dir": "../datasets/nfcorpus/",
        "instruction": {
            "corpus": "",
            "queries": "Given a question, retrieve relevant documents that answer the question"
        },
        "batch_size": 4,
        "file_template": "{split}.jsonl",
        "output": "./nfcorpus/{split}_hugging_face.npz",
        "processor": default_processor
    },
    "scifact": {
        "dir": "../datasets/scifact/",
        "instruction": {
            "corpus": "",
            "queries": "Given a scientific claim, retrieve documents that support or refute the claim"
        },
        "batch_size": 4,
        "file_template": "{split}.jsonl",
        "output": "./SciFACT/{split}_hugging_face.npz",
        "processor": default_processor
    },
    "fiqa": {
        "dir": "../datasets/fiqa/",
        "instruction": {
            "corpus": "Instruct: Given a text related to finance, please describe the queries it may answer and intent of these queries. \n text:",
            "queries": "Instruct: Given a query relevant to finance, please describe the intent of this query. \n query: "
        },
        "batch_size": 4,
        "file_template": "{split}.jsonl",
        "output": "./fiqa/{split}_from_hugging_face.npz",
        "processor": process_fiqa
    }
}

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for datasets.")
    parser.add_argument('--dataset', required=True, choices=list(DATASET_CONFIG.keys()), help="Dataset name")
    parser.add_argument('--split', required=True, help="Data split (e.g., corpus, queries, train, test)")
    args = parser.parse_args()

    config = DATASET_CONFIG[args.dataset]
    model = get_model()
    batch_size = config["batch_size"]

    if isinstance(config["instruction"], dict):
        instruction = config["instruction"].get(args.split, "")
    else:
        instruction = config["instruction"]

    input_file = os.path.join(config["dir"], config["file_template"].format(split=args.split))
    output_file = config["output"].format(split=args.split)

    processor = config.get("processor", default_processor)
    processor(model, input_file, output_file, instruction, batch_size)

if __name__ == '__main__':
    main()
