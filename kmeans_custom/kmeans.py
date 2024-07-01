import numpy as np
from typing import List


class KMeansCustom:
    def __init__(self, n_clusters: int, max_iteration: int = 300) -> None:
        self.n_clusters = n_clusters
        self.max_iteration = max_iteration
        self.centroids = np.array([])

    def euclidean(self, point: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Calculate the Euclidean distance between a point and all centroids.

        Input :
            point : np.ndarray : A single data point.
            centroids : np.ndarray : All centroids.

        Output :
            np.ndarray : Euclidean distance between a point and all centroids.
        """
        return np.sqrt(np.sum((point - centroids) ** 2, axis=1))

    def fit(self, data: np.ndarray) -> None:
        """
        Fit the KMeansCustom model to the data.

        Input :
            data : np.ndarray : Data to fit the KMeansCustom.

        Output :
            None
        """
        min_, max_ = np.min(data, axis=0), np.max(data, axis=0)
        self.centroids = np.array(
            [np.random.uniform(min_, max_) for _ in range(self.n_clusters)]
        )
        iteration = 0
        previous_centroids = None
        while (
            np.not_equal(self.centroids, previous_centroids).any()
            and iteration < self.max_iteration
        ):
            values_per_nearest_centroid = [[] for _ in range(self.n_clusters)]
            for datum in data:
                distances = self.euclidean(datum, self.centroids)
                nearest_centroid_index = np.argmin(distances)
                values_per_nearest_centroid[nearest_centroid_index].append(datum)
            previous_centroids = self.centroids.copy()
            new_centroids = []
            for values in values_per_nearest_centroid:
                if len(values) > 0:
                    new_centroids.append(np.mean(values, axis=0))
                else:
                    new_centroids.append(previous_centroids[len(new_centroids)])
            self.centroids = np.array(new_centroids)
            iteration += 1

    def predict(self, data: np.ndarray) -> List[int]:
        """
        Predict the cluster index for each data point.

        Input :
            data : np.ndarray : Data to predict the cluster index.

        Output :
            List[int] : Cluster index for each data point.
        """
        centroids_index = []
        for datum in data:
            distances = self.euclidean(datum, self.centroids)
            centroid_index = np.argmin(distances)
            centroids_index.append(centroid_index)
        return centroids_index
