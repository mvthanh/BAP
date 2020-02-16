from __future__ import print_function
import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class KnearestNeighbor_Iris:
    def __init__(self):
        iris = datasets.load_iris()
        self.irisX = iris.data
        self.irisY = iris.target

    def split(self, test_size):
        return train_test_split(self.irisX, self.irisY, test_size=test_size)

    def myweight(self, distances):
        sigma = 0.4
        return np.exp(-distances**2/sigma)

    def KnearestNeighbors(self, n_neighbors, test_size, weight = 'uniform'):
        model = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, p = 2) #p=2 mean norm2
        xtrain, xtest, ytrain, ytest = self.split(test_size)
        model.fit(xtrain, ytrain)
        y_pred = model.predict(xtest)
        print("Accuracy of {}NN: {:.2f} %".format(n_neighbors, 100*accuracy_score(ytest, y_pred)))
  
iris = KnearestNeighbor_Iris()
iris.KnearestNeighbors(1, 130, 'distance')
iris.KnearestNeighbors(1, 130, iris.myweight)
    