
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11) #tao so ngau nhien

class KmeansClustering:
    def __init__(self, means, cov, n = 500):
        self.means = means
        self.cov = cov
        self.n = n

    def initpoint(self):
        x = list(range(len(self.means)))
        label = []
        for i in list(range(len(self.means))):
            x[i] = np.random.multivariate_normal(self.means[i], self.cov, self.n)
            #label = label + [i]*n
        X = x[0]
        for i in range(len(x)):
            if (i == 0): continue
            X = np.concatenate((X, x[i]), axis = 0)
        return X
    
    

    def kmeans(self):
        # chon k diem bat ky lam center
        def kmeans_init_centers(X, k):
            return X[np.random.choice(X.shape[0], k, replace=False)]

        # tim nhan cho cac diem trong x 
        def kmeans_assign_labels(X, centers):
            D = cdist(X, centers)
            return np.argmin(D, axis = 1)
            
        # tim center moi
        def kmeans_update_centers(X, labels, K):
            centers = np.zeros((K, X.shape[1]))
            for k in range(K):
                Xk = X[labels == k, :]
                centers[k, :] = np.mean(Xk, axis = 0)
            return centers

        # kiem tra dieu kien dung
        def has_converged(centers, new_centers):
            return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))

        X = self.initpoint()
        k = len(self.means)
        centers = [kmeans_init_centers(X, k)]
        labels = []
        it = 0
        while True:
            labels.append(kmeans_assign_labels(X, centers[-1]))
            new_centers = kmeans_update_centers(X, labels[-1], k)
            if has_converged(new_centers, centers[-1]):
                break
            centers.append(new_centers)
            it += 1
        return (centers, labels, it) 

                
means = [[2,2], [8,3], [3,6], [5,5]]
cov = [[1,0], [0,1]]
N = 500
    
Kmeans = KmeansClustering(means, cov)

(centers, labels, it) = Kmeans.kmeans()
print('Centers found by our algorithm:')
print(centers[-1], it)
