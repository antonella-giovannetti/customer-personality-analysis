import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import silhouette_score

class KMeans:
    def __init__(self, n_clusters, max_iteration=10):
        self.n_clusters = n_clusters
        self.max_iteration = max_iteration

    def euclidean(self, point, centroids):
        return np.sqrt(np.sum((point - centroids) ** 2, axis=1))

    def fit(self, data):
        min_, max_ = np.min(data, axis=0), np.max(data, axis=0)
        self.centroids = np.array([random.uniform(min_, max_) for _ in range(self.n_clusters)])
        iteration = 0
        previous_centroids = None
        while np.not_equal(self.centroids, previous_centroids).any() and iteration < self.max_iteration:
            values_per_nearest_centroid = [[] for _ in range(self.n_clusters)]
            for datum in data:
                distances = self.euclidean(datum, self.centroids)
                nearest_centroid_index = np.argmin(distances)
                values_per_nearest_centroid[nearest_centroid_index].append(datum)
            previous_centroids = self.centroids
            self.centroids = np.array([np.mean(values, axis=0) for values in values_per_nearest_centroid])
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = previous_centroids[i]
            iteration += 1

    def predict(self, data):
        centroids = []
        centroids_index = []
        for datum in data:
            distances = self.euclidean(datum, self.centroids)
            centroid_index = np.argmin(distances)
            centroids.append(self.centroids[centroid_index])
            centroids_index.append(centroid_index)
        return centroids_index


iris = datasets.load_iris()
data = iris.data
target = iris.target
names = iris.feature_names
data, target = shuffle(data, target, random_state=42)

k_values = [2, 3, 4, 5, 6]
silhouette_scores = {}

for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    predictions = kmeans.predict(data)
    
    score = silhouette_score(data, predictions)
    silhouette_scores[k] = score
    
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(data)
    
    plt.figure(figsize=(8, 6))
    for i in range(k):
        points = data_reduced[np.array(predictions) == i]
        plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {i}')
    plt.scatter(pca.transform(kmeans.centroids)[:, 0], pca.transform(kmeans.centroids)[:, 1], marker='X', s=200, c='red', label='Centroids')
    plt.legend()
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'K-Means Clustering on Iris Dataset (k={k})')
    plt.show()
    
for k, score in silhouette_scores.items():
    print(f"Silhouette score for k={k}: {score}")

plt.figure(figsize=(8, 6))
plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different k values')
plt.show()