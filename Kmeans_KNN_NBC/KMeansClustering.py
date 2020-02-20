from __future__ import print_function
import numpy as np
from scipy.spatial.distance import cdist


np.random.seed(11)  # tao so ngau nhien


class KMeansClustering:
    def __init__(self):
        self.centers = []
        self.labels = []

    def kmeans_fit(self, X, k):
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

        self.centers = [kmeans_init_centers(X, k)]
        it = 0
        while True:
            self.labels.append(kmeans_assign_labels(X, self.centers[-1]))
            new_centers = kmeans_update_centers(X, self.labels[-1], k)
            if has_converged(new_centers, self.centers[-1]):
                break
            self.centers.append(new_centers)
            it += 1

    def predict(self):
        return self.centers, self.labels


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

KMeans = KMeansClustering()
KMeans.kmeans_fit(X, 3)
centers, labels = KMeans.predict()
print('Centers found by our algorithm:')
print(centers[-1])

