import numpy as np


class SoftmaxRegression:
    def __init__(self):
        self.W = []

    def fit(self, X, y, c, lr=0.01, nepoches=100, batch_size=10):
        def softmax_function(Z):
            eZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            return eZ / eZ.sum(axis=1, keepdims=True)

        def softmax_grad(X, y, W):
            A = softmax_function(X.dot(W))  # a = softmax(z) voi z = w.T*x
            id0 = range(X.shape[0])
            A[id0, y] -= 1  # E = A - Y voi y la nhan
            return X.T.dot(A) / X.shape[0]  # 1/N*X*E

        W = np.random.randn(X.shape[1], c)
        W_old = W.copy()
        ep = 0
        N = X.shape[0]
        nbatches = int(np.ceil(float(N) / batch_size))
        while ep < nepoches:
            ep += 1
            mix_ids = np.random.permutation(N)
            for i in range(nbatches):
                batch_ids = mix_ids[batch_size * i:min(batch_size * (i + 1), N)]
                X_batch, y_batch = X[batch_ids], y[batch_ids]
                W -= lr * softmax_grad(X_batch, y_batch, W)
            if np.linalg.norm(W - W_old)/W.size < 1e-5:
                break
            W_old = W.copy()
        self.W = W

    def predict(self, data):
        Z = data.dot(self.W)
        prob = []
        for z in Z:
            z = np.exp(z)
            prob.append(z/z.sum())
        return np.argmax(Z, axis=1), np.array(prob)


C, N = 5, 500
means = [[2, 2], [8, 3], [3, 6], [14, 2], [12, 8]]
cov = [[1, 0], [0, 1]]

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X3 = np.random.multivariate_normal(means[3], cov, N)
X4 = np.random.multivariate_normal(means[4], cov, N)

X = np.concatenate((X0, X1, X2, X3, X4), axis=0)
Xbar = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

y = np.asarray([0] * N + [1] * N + [2] * N + [3] * N + [4] * N)

sm = SoftmaxRegression()
sm.fit(X, y, 5)
k, p = sm.predict(np.array(means))
print(k)
print(p)

