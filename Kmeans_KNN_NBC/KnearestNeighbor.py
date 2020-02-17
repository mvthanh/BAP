from __future__ import print_function
import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class KNearestNeighbor_Iris:
    def __init__(self):
        iris = datasets.load_iris()
        self.irisX = iris.data
        self.irisY = iris.target

    def split(self, test_size):
        return train_test_split(self.irisX, self.irisY, test_size=test_size)

    def myweight(self, distances):
        sigma = 0.4
        return np.exp(-distances ** 2 / sigma)

    def KnearestNeighbors(self, weight='uniform'):
        model = neighbors.KNeighborsClassifier(n_neighbors=5, p=2)  # p=2 mean norm2
        xtrain, xtest, ytrain, ytest = self.split(100)
        model.fit(xtrain, ytrain)
        y_pred = model.predict(xtest)
        print("Accuracy of 5NN: {:.2f} %".format(100 * accuracy_score(ytest, y_pred)))


iris = KNearestNeighbor_Iris()
iris.KnearestNeighbors()
iris.KnearestNeighbors('distance')
iris.KnearestNeighbors(iris.myweight)
