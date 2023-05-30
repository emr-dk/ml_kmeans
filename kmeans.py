'''
Implementation of K-Means algorithm
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate data
# Yes, I know that it kinda defeats the purpose
# of this implementation by using sklearn to generate
# the data, but I'm lazy and I don't want to write
# a function to generate data... For now at least
X, y = make_blobs(n_samples=1000, centers=4, cluster_std=0.60, random_state=0)

# Initialize centroids in python
centroids = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])

# Assign each datapoint to closest centroid
# Calculate the distance between each datapoint and each centroid
# and assign the datapoint to the closest centroid
# This is the E-step of the algorithm
distances = np.zeros((X.shape[0], centroids.shape[0]))
for i in range(centroids.shape[0]):
    distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)

# Assign each datapoint to the closest centroid
# This is the M-step of the algorithm
y = np.argmin(distances, axis=1)

# Update centroids
# Calculate the mean of each cluster and assign it as the new centroid
# This is the M-step of the algorithm
for i in range(centroids.shape[0]):
    centroids[i] = np.mean(X[y == i], axis=0)

# Repeat until convergence
# Repeat the E-step and M-step until the centroids don't change
# This is the convergence step of the algorithm
# This is the full implementation of the K-Means algorithm
while True:
    distances = np.zeros((X.shape[0], centroids.shape[0]))
    for i in range(centroids.shape[0]):
        distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)
    y = np.argmin(distances, axis=1)
    for i in range(centroids.shape[0]):
        centroids[i] = np.mean(X[y == i], axis=0)
    if np.allclose(centroids, old_centroids):
        break
    old_centroids = centroids.copy()

# K-Means algorithm
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plot the result
