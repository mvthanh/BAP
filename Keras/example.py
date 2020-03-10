import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# 6 lop du lieu.
class_names = ['buildings', 'sea', 'mountain', 'glacier', 'forest', 'street']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
IMAGE_SIZE = (150, 150)


def load_data():
    datasets = ['../input/intel-image-classification/seg_train/seg_train',
                '../input/intel-image-classification/seg_test/seg_test']
    output = []
    num = []

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
            if (len(num)) == 6:
                continue
            num.append([len(images), class_names[curr_label]])

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')
        output.append((images, labels))

    images = []
    labels = []
    # lấy ngẫu nhiên 3000 hình trong 14000 hình của tập train là validation data
    for k in range(3000):
        i = np.random.randint(14000)
        images.append(output[0][0][i])
        labels.append(output[0][1][i])

    output.append((np.array(images, dtype='float32'), np.array(labels, dtype='int32')))
    return output


def show_chart(num):
    num[5][0] -= num[4][0]
    num[4][0] -= num[3][0]
    num[3][0] -= num[2][0]
    num[2][0] -= num[1][0]
    num[1][0] -= num[0][0]

    classes = []
    val = []

    for i in num:
        classes.append(i[1])
        val.append(i[0])
    # build biểu đồ
    plt.subplots(figsize=(8, 5))
    plt.bar(classes, val, color='blue')
    plt.axis([-1, 6, 0, 3000])
    plt.ylabel('so luong images')
    plt.xlabel('classes')
    plt.title('train data')

    plt.show()


(train_images, train_labels), (test_images, test_labels), (val_images, val_labels), num = load_data()
show_chart(num)

train_images = train_images / 255.0
test_images = test_images / 255.0
val_images = val_images / 255.0

model = Sequential()
model.add(Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
h = model.fit(train_images, train_labels, batch_size=156, epochs=5, validation_data=(val_images, val_labels))
