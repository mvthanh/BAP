import numpy as np


class Multilayer:
    def __init__(self):
        self.W1 = []
        self.W2 = []
        self.b1 = []
        self.b2 = []

    def init_model(self, d0, d1, d2):
        # d0: So chieu cua input
        # d1: So unit cua hidden layer
        # d2: So class can phan loai
        self.W1 = np.random.randn(d0, d1)
        self.b1 = np.zeros(d1)
        self.W2 = np.random.randn(d1, d2)
        self.b2 = np.zeros(d2)

    def fit(self, X, y, eta=1):
        def softmax_function(Z):
            newZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            return newZ / newZ.sum(axis=1, keepdims=True)

        self.init_model(X.shape[1], 100, np.max(y) + 1)  # ??? thay bang x, y
        # eta: learning rate
        for i in range(1000):
            # feedforward

            Z1 = X.dot(self.W1) + self.b1
            A1 = np.maximum(Z1, 0)
            Z2 = A1.dot(self.W2) + self.b2
            A2 = softmax_function(Z2)

            # back propagation
            id0 = range(A2.shape[0])
            A2[id0, y] -= 1
            E2 = A2 / X.shape[0]
            dW2 = np.dot(A1.T, E2)
            db2 = np.sum(E2, axis=0)
            E1 = np.dot(E2, self.W2.T)
            E1[Z1 <= 0] = 0
            dW1 = np.dot(X.T, E1)
            db1 = np.sum(E1, axis=0)

            # GD
            self.W1 -= eta * dW1
            self.W2 -= eta * dW2
            self.b1 -= eta * db1
            self.b2 -= eta * db2

    def predict(self, data):
        Z1 = data.dot(self.W1) + self.b1
        A1 = np.maximum(Z1, 0)
        Z2 = A1.dot(self.W2) + self.b2
        return np.argmax(Z2, axis=1)


N = 100  # number of points per class
d0 = 2  # dimensionality
C = 3  # number of classes
X = np.zeros((d0, N * C))  # data matrix (each row = single example)
y = np.zeros(N * C, dtype='uint8')  # class labels


for j in range(C):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j*4, (j + 1)*4, N) + np.random.randn(N) * 0.2  # theta
    X[:, ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
    y[ix] = j

mul = Multilayer()
mul.fit(X.T, y)
print(mul.predict(np.array([[0, 0], [0, -10], [-10, 0], [10, 0], [0, 10]])))
print(mul.predict(X.T))
