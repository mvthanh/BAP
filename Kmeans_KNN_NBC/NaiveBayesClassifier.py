from __future__ import print_function
import numpy as np
from scipy.sparse import coo_matrix  # for sparse matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score  # for evaluating results


class NaiveBayesClassifier:

    def __init__(self, data_fn, label_fn):
        self.data_fn = data_fn
        self.label_fn = label_fn

    def read_data(self, data_fn, label_fn):
        # read label_fn
        with open('data/' + label_fn) as f:
            content = f.readlines()
        label = [int(x.strip()) for x in content]

        # read data_fn
        with open('data/' + data_fn) as f:
            content = f.readlines()

        # remove '\n' at the end of each line
        content = [x.strip() for x in content]
        dat = np.zeros((len(content), 3), dtype=int)
        for i, line in enumerate(content):
            a = line.split(" ")
            dat[i, :] = np.array([int(a[0]), int(a[1]), int(a[2])])

        # remember to -1 at coordinate since we’re in Python
        data = coo_matrix((dat[:, 2], (dat[:, 0] - 1, dat[:, 1] - 1)), shape=(len(label), 2500))
        return data, label

    def naivebayes(self):
        (train_data, train_label) = self.read_data(self.data_fn, self.label_fn)
        (test_data, test_label) = self.read_data('test-features.txt', 'test-labels.txt')
        clf = MultinomialNB()
        clf.fit(train_data, train_label)
        y_pred = clf.predict(test_data)
        print("Training size = {}, accuracy = {:.2f} %".format(train_data.shape[0],
                                                               accuracy_score(test_label, y_pred) * 100))


train_data_fn = 'train-features.txt'
train_label_fn = 'train-labels.txt'
nbc = NaiveBayesClassifier(train_data_fn, train_label_fn)
nbc.naivebayes()
