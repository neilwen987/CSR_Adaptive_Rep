import os
import argparse
import numpy as np
from tqdm import tqdm

def split_npz(input_path, output_dir, chunk_size=10000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = np.load(input_path)
    img_emb = data['data']      # shape: (N, ...)
    labels  = data['label']     # shape: (N,)

    N = img_emb.shape[0]
    for start in tqdm(range(0, N, chunk_size)):
        end = min(start + chunk_size, N)
        out_path = os.path.join(output_dir, f'chunk_{start}_{end}.npz')
        np.savez_compressed(
            out_path,
            data  = img_emb[start:end],
            label = labels[start:end]
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a .npz file into smaller chunks.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input .npz file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output chunks')
    parser.add_argument('--chunk_size', type=int, default=10000, help='Number of samples per chunk')

    args = parser.parse_args()

    split_npz(
        input_path=args.input_path,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size
    )
