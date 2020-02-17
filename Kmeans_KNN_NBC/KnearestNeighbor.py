from __future__ import print_function
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class KNearestNeighbor_Iris:
    def __init__(self):
        self.model = neighbors.KNeighborsClassifier(n_neighbors=5, p=2)

    def training(self, data, label):
        self.model.fit(data, label)

    def test(self, data):
        return self.model.predict(data)

    def estimate(self, data, label):
        pred = self.test(data)
        print("Accuracy of 5NN: {:.2f} %".format(100 * accuracy_score(label, pred)))


iris = datasets.load_iris()
data = iris.data
label = iris.target

datatrain, datatest, labeltrain, labeltest = train_test_split(data, label, test_size=130)

KNNIris = KNearestNeighbor_Iris()
KNNIris.training(datatrain, labeltrain)
print(KNNIris.test(datatest))
KNNIris.estimate(datatest, labeltest)

