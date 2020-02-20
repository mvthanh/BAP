import numpy as np


class LogisticRegression:
    def __init__(self):
        self.w = []

    def fit(self, X, y, lamda=0.001, eta=0.1, nepoches=2000):
        # lamda: he so ti le thuan cua w, eta: learning rate, nepoches: so vong lap
        def sigmoid(s):
            return 1/(1 + np.exp(-s))

        N, d = X.shape[0], X.shape[1]
        w = w_old = np.random.randn(X.shape[1])
        ep = 0
        while ep < nepoches:
            ep += 1
            mix_ids = np.random.permutation(N)
            for i in mix_ids:
                xi = X[i]
                yi = y[i]
                w = w - eta*((sigmoid(xi.dot(w)) - yi)*xi + lamda*w)
            if np.linalg.norm(w - w_old)/d < 1e-6:
                # dieu kien dung tuong duong dao ham tien ve 0
                break
            w_old = w
        self.w = w

    def predict(self):
        return self.w


X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

Xbar = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
print(Xbar.shape, y)

LR = LogisticRegression()
LR.fit(Xbar, y)
w = LR.predict()

a = np.array([[-1, 0, 0.001, 1, 3, 5, 10]])
b = w[0] + w[1]*a
print(1 / (1 + np.exp(-b)))
