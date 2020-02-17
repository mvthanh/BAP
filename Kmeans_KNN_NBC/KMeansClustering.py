from __future__ import print_function
import numpy as np
from scipy.spatial.distance import cdist


np.random.seed(11)  # tao so ngau nhien


class KMeansClustering:
    def __init__(self, data, k):
        self.data = data
        self.k = k

    def kmeans(self):
        # chon k diem bat ky lam center
        def kmeans_init_centers(X, k):
            return X[np.random.choice(X.shape[0], k, replace=False)]

        # tim nhan cho cac diem trong x 
        def kmeans_assign_labels(X, centers):
            D = cdist(X, centers)
            return np.argmin(D, axis=1)

        """
                Khi su dung khoach cach Euclid              Expected
                [[2.99084705 6.04196062]                    [3,6]
                [1.97563391 2.01568065]                     [2,2]
                [8.03643517 3.02468432]]                    [8,3]

                Voi D = cdist(X, center, 'cosine')          Expected
                [[2.05259401 5.35465357]                    [3,6]
                [2.9293566  3.70465521]                     [2,2]
                [6.71161478 2.54488532]]                    [8,3]

                -> Dung Euclid cho kq chinh xac hon
                """
        # tim center moi
        def kmeans_update_centers(X, labels, K):
            centers = np.zeros((K, X.shape[1]))
            for k in range(K):
                Xk = X[labels == k, :]
                centers[k, :] = np.mean(Xk, axis=0)
            return centers

        # kiem tra dieu kien dung
        def has_converged(centers, new_centers):
            return set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers])

        centers = [kmeans_init_centers(self.data, self.k)]
        labels = []
        it = 0
        while True:
            labels.append(kmeans_assign_labels(self.data, centers[-1]))
            new_centers = kmeans_update_centers(self.data, labels[-1], self.k)
            if has_converged(new_centers, centers[-1]):
                break
            centers.append(new_centers)
            it += 1
        return centers, labels, it


means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
n = 500

x = []
for i in list(range(len(means))):
    x.append(np.random.multivariate_normal(means[i], cov, n))
X = x[0]
for i in range(len(x)):
    if i == 0:
        continue
    X = np.concatenate((X, x[i]), axis=0)

KMeans = KMeansClustering(X, 3)

(centers, labels, it) = KMeans.kmeans()
print('Centers found by our algorithm:')
print(centers[-1])

"""
labels = np.asarray([0]*n + [1]*n + [2]*n).T
def kmeans_display(X, label):
    K = 3
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)

    plt.axis('equal')
    plt.plot()
    plt.show()

kmeans_display(X, labels[-1])"""
