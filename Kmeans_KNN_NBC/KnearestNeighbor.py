from __future__ import print_function
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist


class KNearestNeighbor_Iris:
    def __init__(self):
        self.data = np.array([])
        self.label = np.array([])

    def fit(self, data, label):
        self.data = data
        self.label = label

    def predict(self, data):
        D = cdist(data, self.data)
        labels = []
        for i in range(data.shape[0]):
            k = np.argmin(D[i])
            labels.append(self.label[k])
        return np.array(labels)


iris = datasets.load_iris()
data = iris.data
label = iris.target

datatrain, datatest, labeltrain, labeltest = train_test_split(data, label, test_size=130)

knn = KNearestNeighbor_Iris()
knn.fit(datatrain, labeltrain)
lb = knn.predict(datatest)
print(lb[:30])
print(labeltest[:30])

