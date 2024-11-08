import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Create a random dataset with 3 clusters
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)

# Apply K-Means clustering to find 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels_kmeans = kmeans.labels_

# Apply AgglomerativeClustering to find 3 clusters
agglom = AgglomerativeClustering(n_clusters=3, metric="euclidean", linkage='ward')
agglom.fit(X)
labels_agglo = agglom.labels_

# Reduce the dimensionality using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=[labels_kmeans[i] for i in range(len(labels_kmeans))], cmap='viridis')
plt.scatter(pca.components_[0, :], pca.components_[1, :], c='red', marker='x', s=200)
plt.title("K-Means Clustering")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=[labels_agglo[i] for i in range(len(labels_agglo))], cmap='viridis')
plt.scatter(pca.components_[0, :], pca.components_[1, :], c='red', marker='x', s=200)
plt.title("Agglomerative Clustering")
plt.show()


from sklearn.metrics import silhouette_score
# Calculate silhouette score for K-Means clustering
silhouette_kmeans = silhouette_score(X_pca, labels_kmeans)

# Calculate silhouette score for AgglomerativeClustering
silhouette_agglom = silhouette_score(X_pca, labels_agglo)

print(f"Silhouette Score for K-Means Clustering: {silhouette_kmeans:.3f}")
print(f"Silhouette Score for AgglomerativeClustering: {silhouette_agglom:.3f}")