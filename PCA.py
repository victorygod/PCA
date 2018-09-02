import numpy as np

def PCA(X, k=0, threshold = 0.95):
	mean = np.mean(X, axis = 0)
	X = X - mean
	U, S, V = np.linalg.svd(np.dot(X.T, X))

	if k==0: #Auto find k
		ss = np.sum(S)
		s = S[0]
		while k<S.shape[0] and s/ss<threshold:
			k+=1
			s+=S[k]
	return np.dot(X, U[:,:k]), U[:,:k]

def reconstruct(p, U):
	return np.dot(p, U.T)