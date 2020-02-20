import numpy as np
import matplotlib.pyplot as plt


class PerceptronLearningAlgorithm:
    def __init__(self):
        self.w = []

    def fit(self, X, y):
        def predict(w, X):
            return np.sign(X.dot(w))

        w = np.random.randn(X.shape[1])
        while True:
            pred = predict(w, X)
            mis_index = np.where(np.equal(pred, y) == False)[0]
            num_mis = mis_index.shape[0]
            if num_mis == 0:
                break
            random_id = np.random.choice(mis_index, 1)[0]
            w = w + y[random_id] * X[random_id]
        self.w = w

    def predict(self):
        return self.w


means = [[-1, 0], [1, 0]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)

X = np.concatenate((X0, X1), axis=0)
y = np.concatenate((np.ones(N), -1 * np.ones(N)))

Xbar = np.concatenate((np.ones((2 * N, 1)), X), axis=1)

PLA = PerceptronLearningAlgorithm()
PLA.fit(Xbar, y)
w = PLA.predict()
x = np.array([-4, 4])
y = - w[0]/w[2] - w[1]/w[2]*x
plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
plt.plot(x, y)
plt.show()
