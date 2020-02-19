import numpy as np


class PerceptronLearningAlgorithm:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def perceptron(self):
        def predict(w, X):
            return np.sign(X.dot(w))

        w = np.random.randn(self.X.shape[1])
        while True:
            pred = predict(w, self.X)
            mis_index = np.where(np.equal(pred, self.y) == False)[0]
            num_mis = mis_index.shape[0]
            if num_mis == 0:
                return w
            random_id = np.random.choice(mis_index, 1)[0]
            w = w + self.y[random_id]*self.X[random_id]


means = [[-1, 0], [1, 0]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)

X = np.concatenate((X0, X1), axis=0)
y = np.concatenate((np.ones(N), -1*np.ones(N)))

Xbar = np.concatenate((np.ones((2*N, 1)), X), axis=1)

PLA = PerceptronLearningAlgorithm(Xbar, y)
print(PLA.perceptron())
