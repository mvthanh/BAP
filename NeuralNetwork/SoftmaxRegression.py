import numpy as np


class SoftmaxRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def softmax(self, lr=0.01, nepoches=100, batch_size=10):
        def softmax_stable(Z):
            eZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            return eZ / eZ.sum(axis=1, keepdims=True)

        def softmax_grad(X, y, W):
            A = softmax_stable(X.dot(W))  # a = softmax(z) voi z = w.T*x
            id0 = range[X.shape[0]]
            A[id0, y] -= 1  # = A - Y voi Y[i] la one-hot
            return X.T.dot(A) / X.shape[0]

        W = np.random.radn(self.X.shape[1], 5)
        W_old = W.copy()
        ep = 0
        N = self.X.shape[0]
        nbatches = int(np.ceil(float(N) / batch_size))
        while ep < nepoches:
            ep += 1
            mix_ids = np.random.permutation(N)
            for i in range(nbatches):
                batch_ids = mix_ids[batch_size * i:min(batch_size * (i + 1), N)]
                X_batch, y_batch = self.X[batch_ids], self.y[batch_ids]
                W -= lr * softmax_grad(X_batch, y_batch, W)
                if np.linalg.norm(W - W_old) / W.size < 1e-5:
                    break
                W_old = W.copy()
            return W


C, N = 5, 500
means = [[2, 2], [8, 3], [3, 6], [14, 2], [12, 8]]
cov = [[1, 0], [0, 1]]

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X3 = np.random.multivariate_normal(means[3], cov, N)
X4 = np.random.multivariate_normal(means[4], cov, N)


