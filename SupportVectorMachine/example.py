import numpy as np
import cv2
import os
from sklearn.metrics import accuracy_score


class SVM:
    def __init__(self):
        self.W = []

    def fit(self, data, labels, lr=.000001, batch_size=128, iters=1000):
        def loss_gradient(W, X, y, regular=0.001):
            N = X.shape[1]
            Z = np.dot(W.T, X)

            correct_class = np.choose(y, Z).reshape(N, 1).T
            margins = np.maximum(0, 1 - correct_class + Z)
            margins[y, N-1] = 0
            loss = margins.sum()/N
            loss += 0.5*regular*np.sum(W*W)

            F = (margins > 0).astype(int)
            F[y, N-1] = np.sum(-F, axis=0)
            dW = X.dot(F.T)/N + regular*W

            return loss, dW

        W = np.random.randn(data.shape[0], 6)
        for it in range(iters):
            id_rand = np.random.choice(data.shape[1], batch_size)

            X_batch = data[:, id_rand]
            y_batch = labels[id_rand]
            loss, dW = loss_gradient(W, X_batch, y_batch)
            W -= lr*dW
            if it % 100 == 1:
                print(it, loss)
                self.W = W
                res = svm.predict(X_batch[:X_batch.shape[0]-1][:].T)
                print("Accuracy of SVM: ", (100 * accuracy_score(y_batch, res)), '%')

        self.W = W

    def predict(self, data):
        data = np.concatenate((data, np.ones((data.shape[0], 1))), axis=1).T
        return np.argmax(np.dot(self.W.T, data), axis=0)

    def get_res(self):
        w = self.W[: self.W.shape[0]-1, :]
        w = w.T
        for it in range(6):
            k = w[it]
            k = np.reshape(k, (150, 150, -1))
            cv2.imshow("img" + str(it), k)
        cv2.waitKey(0)


class_names = ['buildings', 'sea', 'mountain', 'glacier', 'forest', 'street']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
IMAGE_SIZE = (150, 150)


def load_data():
    datasets = ['./input/test/seg_train/seg_train',
                './input/test/seg_test/seg_test']
    output = []

    # Iterate through training and test sets
    for dataset in datasets:
        images = []
        labels = []

        print("Loading {}".format(dataset))

        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            curr_label = class_names_label[folder]

            # Iterate through each image in our folder
            i = 0
            for file in os.listdir(os.path.join(dataset, folder)):
                i += 1
                if i > 500:
                    break
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)

                # Open and resize the img
                curr_img = cv2.imread(img_path)
                curr_img = cv2.resize(curr_img, IMAGE_SIZE)

                # Append the image and its corresponding label to the output
                images.append(curr_img)
                labels.append(curr_label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')
        output.append((images, labels))

    return output


(train_images, train_labels), (test_images, test_labels) = load_data()
train_images /= 255.0
test_images /= 255.0

train_images = np.reshape(train_images, (train_images.shape[0], -1))
train_images = np.concatenate((train_images, np.ones((train_images.shape[0], 1))), axis=1).T

test_images = np.reshape(test_images, (test_images.shape[0], -1))

svm = SVM()
svm.fit(train_images, train_labels)

res = svm.predict(test_images)
print("Accuracy of SVM on data test: ", (100*accuracy_score(test_labels, res)), '%')

#svm.get_res()
