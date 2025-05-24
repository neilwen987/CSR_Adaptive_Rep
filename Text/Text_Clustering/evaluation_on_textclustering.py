import argparse
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.optimize import linear_sum_assignment

def cluster_accuracy(y_true, y_pred):
    """
    Calculate Top-1 clustering accuracy using the Hungarian algorithm.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    D = max(y_pred.max(), y_true.max()) + 1
    confusion = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        confusion[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(confusion.max() - confusion)
    total_correct = confusion[row_ind, col_ind].sum()
    return total_correct / len(y_pred)

def main(args):
    data = np.load(args.embedding_path)
    X = data['data']
    y = data['label']

    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        batch_size=args.batch_size,
        n_init=args.n_init
    )
    kmeans.fit(X)
    y_pred = kmeans.labels_

    acc = cluster_accuracy(y, y_pred)
    print(f"Top-1 Clustering Accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniBatchKMeans clustering and cluster accuracy calculation.")
    parser.add_argument("--embedding_path", type=str, required=True, help="Path to the embedding .npz file")
    parser.add_argument("--n_clusters", type=int, default=26, help="Number of clusters")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--n_init", type=str, default="auto", help="Number of initializations ('auto' or int)")
    args = parser.parse_args()
    main(args)
