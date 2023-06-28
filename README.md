# Introduction
This is an implementation of the kmeans algorithm in Python. 

## Mathematical background
The kmeans algorithm is an algorithm for clustering data. The algorithm takes as input a set of data points and a number k and outputs k clusters. The algorithm works as follows:
1. Initialize _k_ centroids
2. Assign each data point to the closest centroid
3. Update the centroids by taking the mean of the data points assigned to each centroid
4. Repeat steps 2 and 3 until the centroids do not change or the maximum number of iterations is reached

The mathematical notation for the kmeans model is as follows:
$$\argmin_{\mu_1, \ldots, \mu_k} \sum_{i=1}^n \min_{j=1, \ldots, k} \|\mathbf{x}_i - \mu_j\|^2$$
where $\mathbf{x}_i$ is the $i$ th data point and $\mu_j$ is the $j$ th centroid.

We will be implementing the algorithm using the expectation maximization algorithm. The expectation maximization algorithm is an algorithm for finding the maximum likelihood estimate of a model. The algorithm works as follows:
1. Initialize the parameters
2. Repeat until convergence:
    1. E-step: compute the expected value of the log likelihood function
    2. M-step: maximize the expected value of the log likelihood function


## Implementation
The implementation of the kmeans algorithm is in the file kmeans.py.
