import numpy as np
from numpy._core.defchararray import isnumeric
from scipy.linalg import svd as SVD


DEBUG_FLAG_PRINT = False
DEBUG_FLAG_TEST = False
APPROXIMATION_FLAG = True


class MachineLearningTemplate:
    """
    A template for machine learning models with methods for training, prediction,
    and accessing model parameters and hyperparameters.

    GLOBAL NO_LABEL = -1

    """

    NO_LABEL = -1

    def __init__(self, paramsAssigned: bool, hyperParams, learned):
        """
        Initializes with hyperparameters, learned parameters, and an assignment flag.

        Args:
            paramsAssigned (bool): Flag indicating if parameters are initialized.
            hyperParams: Model hyperparameters.
            learned: Model learned parameters.

        Functions:
          train --> Implement in subclasses
          predict --> Implement in subclasses
          getParamsAssigned --> bool
          getHyperParameters --> dict of hyperparameters
        """
        self.paramsAssigned = paramsAssigned
        self.hyperParams = hyperParams
        self.learned = learned

    def train(self, hyperParams: dict, X, y):
        pass

    def predict(self, X) -> list:
        return []

    def getParamsAssigned(self) -> bool:
        return self.paramsAssigned

    def getHyperParameters(self) -> dict:
        return self.hyperParams


class K_Means(MachineLearningTemplate):
    """
    A K-Means clustering implementation inheriting from MachineLearningTemplate.

    Attributes:
        paramsAssigned (bool): Indicates if the model parameters have been assigned.
        hyperParams (dict): Dictionary of hyperparameters like K, max_iter, and tau.
        learned (np.ndarray): The centroids learned during training.
    """

    def __init__(self, paramsAssigned: bool, hyperParams, learned):
        """
        Initialize the K-Means model.

        Args:
            paramsAssigned (bool): Indicates if model parameters are assigned.
            hyperParams (dict): Hyperparameters like K, max_iter, and tau.
            learned (np.ndarray): Centroids to be learned during training.
        """
        super().__init__(paramsAssigned, hyperParams, learned)

    def train(self, hyperParams: dict, X: np.ndarray, y=None):
        """
        Train the K-Means model.

        Args:
            hyperParams (dict): Dictionary containing K, max_iter, and tau (threshold).
            X (np.ndarray): The dataset as a NumPy array.
            y (Optional): Ignored, included for compatibility.

        Raises:
            ValueError: If K, max_iter, or tau values are invalid.
        """

        # Sanity checks
        K = hyperParams.get("K")
        if K is not None:
            if type(K) is not int or K <= 0:
                raise ValueError("K must be of type int and be > 0")
        else:
            raise ValueError("K must be specified")

        max_iter = hyperParams.get("max_iter")
        if max_iter is not None:
            if type(max_iter) is not int or max_iter <= 0:
                raise ValueError("max_iter must of of type int and be > 0")
        else:
            raise ValueError("max_iter must be specified")

        threshold = hyperParams.get("tau")
        if threshold is not None:
            if threshold < 0:
                raise ValueError("threshold must be >= 0")
        else:
            threshold = 1e-05

        # Initialize centroids as zeros with shape (K, number of features).
        self.learned = np.zeros((K, len(X[0])))
        copied = np.copy(self.learned)

        # Append a cluster label column to the data and initialize clusters randomly.
        X_with_clusters = np.hstack((X, np.zeros((X.shape[0], 1), dtype=int)))
        K_entries = np.zeros(K, dtype=int)
        for row_idx in range(X_with_clusters.shape[0]):
            rand_k = np.random.randint(0, K)
            X_with_clusters[row_idx, -1] = rand_k
            K_entries[rand_k] += 1

        convergence = False
        iters = 0
        print("Training")
        while not convergence and iters < max_iter:
            copied = np.copy(self.learned)
            self.learned = np.zeros((K, X.shape[1]))

            # Accumulate features for each cluster.
            for i in range(X_with_clusters.shape[0]):
                features = X_with_clusters[i, :-1]
                cluster = int(X_with_clusters[i, -1])
                self.learned[cluster] += features

            # Compute the average for each cluster to update centroids.
            for k in range(K):
                avg = self.learned[k] / (1.0 * K_entries[k])
                self.learned[k] = avg

            # Reassign points to the nearest centroid.
            for i in range(X_with_clusters.shape[0]):
                min_distance = float("inf")
                for k in range(K):
                    features = X_with_clusters[i, :-1]
                    centroid = self.learned[k]
                    distance = np.linalg.norm(features - centroid)
                    if distance < min_distance:
                        min_distance = distance
                        X_with_clusters[i, -1] = k

            # Check convergence by comparing centroids.
            convergence = True
            for k in range(K):
                diff = np.linalg.norm(self.learned[k] - copied[k])
                convergence = convergence and threshold > diff
                if convergence:
                    print(
                        f"converged diff={diff} , learned {self.learned[k]}, copied {copied[k]}, threshold={threshold}"
                    )
            print(f"Iteration={iters}")
            iters += 1
        self.paramsAssigned = True

    def predict(self, X):
        """
        Predict the closest cluster for each data point.

        Args:
            X (np.ndarray): Dataset to be clustered.

        Returns:
            list: Cluster indices for each data point.

        Raises:
            AssertionError: If the model is not trained before prediction.
        """
        if not self.paramsAssigned:
            raise AssertionError(
                "Please Call train before calling the predict  function"
            )

        z = []
        for row in X:
            min_distance = float("inf")
            closest = 0
            # Find the nearest cluster centroid.
            for k in range(self.getHyperParameters()["K"]):
                diff = np.linalg.norm(row - self.learned[k])
                if diff < min_distance:
                    min_distance = diff
                    closest = k

            z.append(closest)

        return z


class K_Means_Classifier(K_Means):
    """
    A supervised K-Means classifier that extends the K-Means clustering algorithm.
    Allows labeled data to influence the clustering process.

    Attributes:
        paramsAssigned (bool): Indicates if the model parameters have been assigned.
        hyperParams (dict): Dictionary of hyperparameters such as K, max_iter, and tau.
        learned (np.ndarray): Centroids learned during training.
    """

    def __init__(self, paramsAssigned: bool, hyperParams, learned):
        """
        Initialize the K-Means Classifier model.

        Args:
            paramsAssigned (bool): Indicates if model parameters are assigned.
            hyperParams (dict): Hyperparameters like K, max_iter, and tau.
            learned (np.ndarray): Centroids to be learned during training.
        """
        super().__init__(paramsAssigned, hyperParams, learned)

    def train(self, hyperParams: dict, X: np.ndarray, y=None):
        """
        Train the K-Means Classifier model.

        Args:
            hyperParams (dict): Dictionary containing K, max_iter, and tau (threshold).
            X (np.ndarray): The dataset as a NumPy array.
            y (Optional): Labels for supervised learning. If None, the algorithm is unsupervised.

        Raises:
            ValueError: If K, max_iter, or tau values are invalid.
        """
        K = hyperParams.get("K")
        if K is not None:
            if type(K) is not int or K <= 0:
                raise ValueError("K must be of type int and be > 0")
        else:
            raise ValueError("K must be specified")

        max_iter = hyperParams.get("max_iter")
        if max_iter is not None:
            if type(max_iter) is not int or max_iter <= 0:
                raise ValueError("max_iter must of of type int and be > 0")
        else:
            raise ValueError("max_iter must be specified")

        threshold = hyperParams.get("tau")
        if threshold is not None:
            if threshold < 0:
                raise ValueError("threshold must be >= 0")
        else:
            threshold = 1e-05

        # New vector to store labeled data flags
        labeled_samples = np.zeros(X.shape[0], dtype=bool)
        if y is not None:
            labeled_samples = np.ones(X.shape[0], dtype=bool)

        # Initialize centroids as zeros with shape (K, number of features).
        self.learned = np.zeros((K, len(X[0])))
        copied = np.copy(self.learned)
        X_with_clusters = np.hstack((X, np.zeros((X.shape[0], 1), dtype=int)))
        K_entries = np.zeros(K, dtype=int)
        for row_idx in range(X_with_clusters.shape[0]):
            # Assign labeled samples to their respective clusters.
            if y is not None:
                if labeled_samples[row_idx]:
                    X_with_clusters[row_idx, -1] = y[row_idx]
                    K_entries[y[row_idx]] += 1

                # Assign unlabeled samples a temporary "no label" cluster.
                else:
                    rand_k = self.NO_LABEL
                    X_with_clusters[row_idx, -1] = rand_k
                    K_entries[rand_k] += 1

        convergence = False
        iters = 0
        print("Training")
        while not convergence and iters < max_iter:
            copied = np.copy(self.learned)
            self.learned = np.zeros((K, X.shape[1]))

            # Accumulate features for each cluster.
            for i in range(X_with_clusters.shape[0]):
                if int(X_with_clusters[i, -1]) == -1:
                    continue
                features = X_with_clusters[i, :-1]
                cluster = int(X_with_clusters[i, -1])
                self.learned[cluster] += features

            # Compute the new centroid positions.
            for k in range(K):
                avg = self.learned[k] / (1.0 * K_entries[k])
                self.learned[k] = avg

            # Reassign unlabeled samples to the nearest centroid.
            for i in range(X_with_clusters.shape[0]):
                if labeled_samples[i]:
                    continue
                min_distance = float("inf")
                for k in range(K):
                    features = X_with_clusters[i, :-1]
                    centroid = self.learned[k]
                    distance = np.linalg.norm(features - centroid)
                    if distance < min_distance:
                        min_distance = distance
                        X_with_clusters[i, -1] = k

            convergence = True
            for k in range(K):
                diff = np.linalg.norm(self.learned[k] - copied[k])
                convergence = convergence and threshold > diff
                if convergence:
                    print(
                        f"converged diff={diff} , learned {self.learned[k]}, copied {copied[k]}, threshold={threshold}"
                    )
            print(f"Iteration={iters}")
            iters += 1
        self.paramsAssigned = True

    def predict(self, X):
        """
        Predict the closest cluster for each data point.

        Args:
            X (np.ndarray): Dataset to be clustered.

        Returns:
            list: Cluster indices for each data point.

        Raises:
            AssertionError: If the model is not trained before prediction.
        """
        if not self.paramsAssigned:
            raise AssertionError(
                "Please Call train before calling the predict  function"
            )

        z = []
        for row in X:
            min_distance = float("inf")
            closest = 0

            # Find the nearest cluster centroid.
            for k in range(self.getHyperParameters()["K"]):
                diff = np.linalg.norm(row - self.learned[k])
                if diff < min_distance:
                    min_distance = diff
                    closest = k

            z.append(closest)

        return z


class PrincipalComponentAnalysis(MachineLearningTemplate):
    """
    A Principal Component Analysis (PCA) implementation for dimensionality reduction.

    Attributes:
        paramsAssigned (bool): Indicates if the model parameters have been assigned.
        hyperParams (dict): Dictionary of hyperparameters such as K (number of components).
        learned (np.ndarray): Principal components learned during training.
    """

    def __init__(self, paramsAssigned: bool, hyperParams, learned):
        """
        Initialize the PCA model.

        Args:
            paramsAssigned (bool): Indicates if model parameters are assigned.
            hyperParams (dict): Hyperparameters like K (number of principal components).
            learned (np.ndarray): Principal components to be learned during training.
        """
        super().__init__(paramsAssigned, hyperParams, learned)

    def train(self, hyperParams: dict, X, y=None):
        """
        Train the PCA model by computing principal components.

        Args:
            hyperParams (dict): Dictionary containing K (number of components to retain).
            X (np.ndarray): The dataset as a NumPy array.
            y (Optional): Not used in PCA, included for template compatibility.

        Raises:
            ValueError: If K value is invalid or not specified.
        """
        K = hyperParams.get("K")
        if K is not None:
            if K <= 0 or K >= min(X.shape[1], X.shape[0]):
                raise ValueError(
                    f"K value {K} must be positve and less than col {X.shape[1]} and row {X.shape[0]}"
                )
        else:
            raise ValueError("K must be specified")

        means = np.mean(X, axis=0)
        std_devs = np.std(X, axis=0)
        # Standardize X
        X_standardized = self.standardize_data(X, means, std_devs)

        U, S, V = SVD(X_standardized, full_matrices=True)

        # Select the appropriate matrix for eigenvectors based on matrix dimensions
        eigens = np.transpose(U) if U.shape[1] == X_standardized.shape[1] else V
        if __debug__:
            print("U") if U.shape[1] == X_standardized.shape[1] else print("V")
            print(f"Eigen Shape {eigens.shape}")

        # Store the top K principal components
        self.learned = eigens[:K]
        if __debug__:
            print(f"Learned {self.learned.shape}")
        self.paramsAssigned = True

    def predict(self, X):
        """
        Perform dimensionality reduction and reconstruction of the input data.

        Args:
            X (np.ndarray): Dataset to be reduced and reconstructed.

        Returns:
            np.ndarray: Reconstructed data after dimensionality reduction.

        Raises:
            AssertionError: If the model is not trained before prediction.
        """
        if not self.paramsAssigned:
            raise AssertionError("Please call train before calling predict")

        means = np.mean(X, axis=0)
        std_devs = np.std(X, axis=0)

        X_standardized = self.standardize_data(X, means, std_devs)

        # Project data onto principal components (dimensionality reduction)
        X_compressed_standardized = np.matmul(
            self.learned, np.transpose(X_standardized)
        )

        if __debug__:
            print(f"X_compressed_standardized shape: {X_compressed_standardized.shape}")

        # Reconstruct data from reduced representation
        X_decompressed_standardized = np.matmul(
            np.transpose(self.learned), X_compressed_standardized
        )

        # Ensure correct matrix orientation
        if X_decompressed_standardized.shape != X_standardized.shape:
            X_decompressed_standardized = np.transpose(X_decompressed_standardized)

        return self.undo_standardize_data(X_decompressed_standardized, means, std_devs)

    def undo_standardize_data(self, X_standardized, means, std_devs):
        """
        Reverse the standardization process to return data to its original scale.

        Args:
            X_standardized (np.ndarray): Standardized data.
            means (np.ndarray): Mean values used for standardization.
            std_devs (np.ndarray): Standard deviation values used for standardization.

        Returns:
            np.ndarray: Data restored to its original scale.
        """
        if __debug__:
            print(
                f"Performing Undo Standardization with inputs mean {means}, std {std_devs}"
            )
        std_devs_diag = np.diag(std_devs)
        X_scaled = np.matmul(X_standardized, std_devs_diag)

        mean_tiled = np.tile(means, (X_standardized.shape[0], 1))
        if __debug__:
            print(f"Mean tiled shape In Undo {mean_tiled.shape}")

        return X_scaled + mean_tiled

    def standardize_data(self, X, means, std_devs):
        """
        Standardize the input data by centering and scaling.

        Args:
            X (np.ndarray): Input data to be standardized.
            means (np.ndarray): Mean values to center the data.
            std_devs (np.ndarray): Standard deviation values to scale the data.

        Returns:
            np.ndarray: Standardized data with zero mean and unit variance.
        """
        if __debug__:
            print(
                f"Performing Standardization with inputs means {means}, std {std_devs}"
            )

        mean_tiled = np.tile(means, (X.shape[0], 1))
        X_centered = X - mean_tiled
        std_devs_inverted = np.power(std_devs, -1)
        std_devs_iverted_diag = np.diag(std_devs_inverted)
        X_standardized = np.matmul(X_centered, std_devs_iverted_diag)

        return X_standardized
