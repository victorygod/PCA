# Principal component analysis (PCA) 

Algorithm flow:

1. Decentralization (X = X - mean)
2. Get covariance matrix (cov = np.dot(X.T, X))
3. SVD (U, S, V = np.linalg.svd(cov))

The first several colomns of U is the Dimensionality reduction matrix.

4. Auto-determin k: Try to find the smallest k which makes sum(S[0:k])/sum(S) < threshold. Threshold usually equals 0.95 or 0.99.
5. output data after reducing dimension. (return np.dot(X, U[:,:k]))
6. To reconstruct data from the compressed data, just np.dot(compressed_data, U[:,:k].T).
