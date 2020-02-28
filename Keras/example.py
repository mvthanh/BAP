import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn

sn.set(font_scale=1.4)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2

# Here's our 6 categories that we have to classify.
class_names = ['buildings', 'sea', 'mountain', 'glacier', 'forest', 'street']

class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)
# nb_classes = 6

IMAGE_SIZE = (150, 150)


def load_data():
    datasets = ['../input/intel-image-classification/seg_train/seg_train', '../input/intel-image-classification/seg_test/seg_test']
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
            for file in os.listdir(os.path.join(dataset, folder)):
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

    images = []
    labels = []
    for k in range(3000):
        i = np.random.randint(14000)
        images.append(output[0][0][i])
        labels.append(output[0][1][i])

    output.append((np.array(images, dtype='float32'), np.array(labels, dtype='int32')))

    return output


"""def display_examples(class_names, images, labels):
    
        #Display 25 images from the images array with its corresponding labels
    

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    for i in range(50):
        ii = np.random.randint(14000)
        plt.subplot(5, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[ii], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[ii]])
    plt.show()"""


(train_images, train_labels), (test_images, test_labels), (val_images, val_labels) = load_data()

#display_examples(class_names, train_images, train_labels)

print("Number of training examples: " + str(train_labels.shape[0]))
print("Number of testing examples: " + str(test_labels.shape[0]))
print("Each image is of size: " + str(train_images.shape[1:]))

train_images = train_images / 255.0
test_images = test_images / 255.0

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


model = Sequential()
model.add(Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(150, 150, 3)))
model.add(Conv2D(16, (3, 3), padding='valid', activation='relu'))
model.add(Conv2D(16, (3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

# sgd = optimizers.SGD(lr=0.05)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
h = model.fit(train_images, train_labels, batch_size=256, epochs=5, validation_data=(val_images, val_labels))
