import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score

TASK_LIST = ["Banking777", "mtop_intent", "tweet_sentiment_extraction"]

def load_data(path):
    data = np.load(path)
    X = data['data']
    y = data['label']
    return X, y

def main(args):
    X_train, y_train = load_data(args.train_embedding_path)
    X_test, y_test = load_data(args.test_embedding_path)

    clf = LogisticRegression(
        n_jobs=args.n_jobs,
        max_iter=args.max_iter,
        verbose=args.verbose
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "average_precision": average_precision_score(y_test, clf.predict_proba(X_test), average="weighted") \
            if len(np.unique(y_test)) == 2 or len(np.unique(y_test)) > 2 else None
    }

    print("Scores:", scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a Logistic Regression classifier on embedding data.")
    parser.add_argument("--train_embedding_path", type=str, required=True, help="Path to the train embedding .npz file")
    parser.add_argument("--test_embedding_path", type=str, required=True, help="Path to the test embedding .npz file")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of CPU cores to use")
    parser.add_argument("--max_iter", type=int, default=2000, help="Maximum number of iterations for the solver")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity level")
    args = parser.parse_args()
    main(args)
