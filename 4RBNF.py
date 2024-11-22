import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(0)
X = np.random.rand(100, 2)
y = np.sin(X[:, 0]) + np.cos(X[:, 1])

# Define the radial basis function (RBF)
def rbf(x, c, s):
    return np.exp(-np.linalg.norm(x - c) * 2 / (2 * s * 2))

# Choose centers using k-means
kmeans = KMeans(n_clusters=10)
kmeans.fit(X)
centers = kmeans.cluster_centers_

# Calculate the spread parameter (sigma)
d_max = np.max(cdist(centers, centers, 'euclidean'))
sigma = d_max / np.sqrt(2 * len(centers))

# Compute the RBF layer output
R = np.zeros((X.shape[0], len(centers)))
for i in range(X.shape[0]):
    for j in range(len(centers)):
        R[i, j] = rbf(X[i], centers[j], sigma)

# Compute the output weights
W = np.dot(np.linalg.pinv(R), y)

# Define the RBF network prediction function
def rbf_network(X, centers, sigma, W):
    R = np.zeros((X.shape[0], len(centers)))
    for i in range(X.shape[0]):
        for j in range(len(centers)):
            R[i, j] = rbf(X[i], centers[j], sigma)
    return np.dot(R, W)

# Make predictions
y_pred = rbf_network(X, centers, sigma, W)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse}")

