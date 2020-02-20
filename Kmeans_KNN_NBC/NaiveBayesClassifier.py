from __future__ import print_function
import numpy as np

'''Văn bản Nội dung
Tập huấn luyện
Kiểm thử
d1      hanoi pho chaolong hanoi       B    
d2      hanoi buncha pho omai          B 
d3      pho banhgio omai               B 
d4      saigon hutiu banhbo pho        N
d5      hanoi hanoi buncha hutiu       ?
'''


class NaiveBayesClassifier:
    def __init__(self):
        self.dictionary = []
        self.label = []
        self.probable = []

    def fit(self, data, labels):
        labels.sort()
        probable = []
        for lb in labels:
            if len(self.label) == 0:
                self.label.append(lb)
                probable.append(1)
                continue
            if lb == self.label[-1]:
                probable[-1] += 1
                continue
            else:
                self.label.append(lb)
                probable.append(1)
        self.probable = np.array(probable)
        self.probable = self.probable/self.probable.sum()

        for lb in self.label:
            c_data = data[labels == lb]
            sum = np.sum(c_data, axis=0)
            self.dictionary.append((sum+1)/(sum.size+sum.sum()))

    def predict(self, data):
        pred = []
        prob = []
        for dt in data:
            result = []
            for i in range(len(self.dictionary)):
                p = self.probable[i]
                for j in range(dt.size):
                    p *= self.dictionary[i][j]**dt[j]
                result.append(p)
            pred.append(self.label[np.argmax(result)])
            result = np.array(result)
            result = result/result.sum()
            prob.append(np.max(result))
        return pred, prob


nbc = NaiveBayesClassifier()

d1 = [2, 1, 1, 0, 0, 0, 0, 0, 0]
d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]
train_data = np.array([d1, d2, d3, d4])
label = np.array(['B', 'B', 'B', 'N'])

d5 = np.array([[2, 0, 0, 1, 0, 0, 0, 1, 0]])
d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])

nbc.fit(train_data, label)
print(nbc.predict(d5))

