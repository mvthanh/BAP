import numpy as np


class GradientDescent:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def myGD(self, w_init, grad, eta=1):
        w = [w_init]
        for it in range(100):
            w_new = w[-1] - eta * grad(w[-1], self.X, self.y)
            if np.linalg.norm(grad(w_new, self.X, self.y)) / len(w_new) < 1e-3:
                break
            w.append(w_new)
        return w, it

    def GDMomentum(self, theta_init, grad, eta=1, gama=0.9):
        theta = [theta_init]
        v = np.zeros_like(theta_init)
        for it in range(100):
            v_new = gama * v + eta * grad(theta[-1], self.X, self.y)
            theta_new = theta[-1] - v_new
            if np.linalg.norm(grad(theta_new, self.X, self.y)) / np.array(theta_init).size < 1e-3:
                break
            v = v_new
            theta.append(theta_new)
        return theta, it

    def SGD(self, theta_init, grad, eta=1, gama=0.9):
        theta = [theta_init]
        v = np.zeros_like(theta_init)
        for it in range(100):
            v_new = gama * v + eta * grad(theta[-1] - gama*v, self.X, self.y)
            theta_new = theta[-1] - v_new
            if np.linalg.norm(grad(theta_new, self.X, self.y)) / np.array(theta_init).size < 1e-3:
                break
            v = v_new
            theta.append(theta_new)
        return theta, it


N = 1000
X = np.random.rand(N)
y = 4 + 3 * X + .5 * np.random.randn(N)
y = y.reshape(-1, 1)
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X.reshape(-1, 1)), axis=1)
w_init = np.array([[2], [1]])

print(Xbar.shape, y.shape, w_init.shape)
eta = 1
gama = 0.9


def grad(w, Xbar, y):
    N = Xbar.shape[0]
    return 1 / N * Xbar.T.dot(Xbar.dot(w) - y)


GD = GradientDescent(Xbar, y)
w, it = GD.myGD(w_init, grad)
print(it)
print(w[-1].T)

w, it = GD.GDMomentum(w_init, grad)
print(it)
print(w[-1].T)

w, it = GD.SGD(w_init, grad)
print(it)
print(w[-1].T)
