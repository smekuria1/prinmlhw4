import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, fetch_california_housing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from MLClasses import K_Means, K_Means_Classifier, PrincipalComponentAnalysis


def main():
    """Main function to run the model training and evaluation."""
    parser = argparse.ArgumentParser(description="Train K-Means")
    # Model selection
    parser.add_argument(
        "--model",
        choices=["k-means", "k-class", "pca"],
        required=True,
        help="Choose the model to train",
    )

    parser.add_argument(
        "--blob_count",
        type=int,
        default=10,
        help="Choose blob count for make_blobs",
    )

    # Number of clusters for K-Means
    parser.add_argument(
        "--n_clusters", type=int, default=10, help="Number of clusters for K-Means"
    )

    # Hyperparameters for custom models
    parser.add_argument(
        "--max_iter", type=int, default=10, help="Maximum iterations for K-Means"
    )

    args = parser.parse_args()
    hyperParams = {
        "K": args.n_clusters,
        "max_iter": args.max_iter,
        "tau": 0.0001,
    }

    X_train, true_labels = make_blobs(150, hyperParams["K"], random_state=42)
    if args.model == "k-means":
        k_means_model(hyperParams, X_train)
    elif args.model == "pca":
        print("---------Using PCA with California Housing--------")
        dataset = fetch_california_housing()
        X, y = dataset.data, dataset.target
        pca_model(hyperParams, X, y)
    else:
        k_class_model(hyperParams, X_train, true_labels)


def pca_model(hyperParams: dict, X_train, y=None):
    model = PrincipalComponentAnalysis(False, hyperParams, None)
    print(f"------Dataset Shape {X_train.shape}----------")

    model.train(hyperParams, X_train, y)
    sklearn_pca = PCA(hyperParams["K"])
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_Scaled = scaler.transform(X_train)

    sklearn_pca.fit(X_Scaled)
    result = model.predict(X_train)
    sk_result = sklearn_pca.transform(X_Scaled)
    if __debug__:
        print(f"Transformed Data {result[:4]}")
        print("------------")
    print(f"Prediction Results {result.shape}")
    print(f"SK learn results {sk_result.shape}")


def k_means_model(hyperParams: dict, X_train, y=None):
    model = K_Means(False, hyperParams, None)
    print(f"Dataset Shape {X_train.shape}")

    skmeans = KMeans(hyperParams["K"])
    skmeans.fit(X_train)
    model.train(hyperParams, X_train)
    model.predict(X_train)
    skmeans.predict(X_train)

    print(f"Sk-Learn Centroids \n {skmeans.cluster_centers_}")
    print(f"Model Centroids \n {model.learned}")


def k_class_model(hyperParams: dict, X_train, y=None):
    true_labels = y
    model = K_Means_Classifier(False, hyperParams, None)
    print(f"Dataset Shape {X_train.shape}")

    skmeans = KMeans(hyperParams["K"])
    skmeans.fit(X_train)
    model.train(hyperParams, X_train, true_labels)
    preds = model.predict(X_train)
    skmeans.predict(X_train)

    print(f"Accuracy = {accuracy_score(true_labels, preds)*100}%")

    print(f"Sk-Learn Centroids \n {skmeans.cluster_centers_}")
    print(f"Model Centroids \n {model.learned}")


if __name__ == "__main__":
    main()
