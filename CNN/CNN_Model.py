import numpy as np
import cv2


class CNN_Model:
    def __init__(self):
        self.model = []

    def flatten(self, data):
        new_arr = np.array(data)
        return np.reshape(new_arr, (-1))

    def maxpooling2d(self, pool_size, data):
        input_shape = data.shape
        l = len(input_shape)
        if l == 2:
            input_shape = (input_shape[0], input_shape[1], 1)
        result = []
        row = 0
        while row < input_shape[0]:
            col = 0
            res_col = []
            while col < input_shape[1]:
                res_point = []
                for layer in range(input_shape[2]):
                    if l == 2:
                        a = data[row: min(row + pool_size[0], input_shape[0]),
                            col: min(col + pool_size[1], input_shape[1])]
                    else:
                        a = data[row: min(row + pool_size[0], input_shape[0]),
                            col: min(col + pool_size[1], input_shape[1]), layer]
                    res_point.append(np.max(a))
                res_col.append(res_point)
                col += pool_size[1]
            result.append(res_col)
            row += pool_size[0]
        return result

    def conv2d(self, kernel, data):
        input_shape = data.shape
        filter_size = kernel.shape
        l = len(input_shape)
        if l == 2:
            input_shape = (input_shape[0], input_shape[1], 1)
        sum_res = np.zeros((input_shape[0] - filter_size[0] + 1, input_shape[1] - filter_size[1] + 1))
        row = 0
        while (row + filter_size[0]) < input_shape[0]:
            col = 0
            while (col + filter_size[1]) < input_shape[1]:

                for layer in range(input_shape[2]):
                    if l == 2:
                        a = data[row: row + filter_size[0], col: col + filter_size[1]]
                    else:
                        a = data[row: row + filter_size[0], col: col + filter_size[1], layer]
                    res = np.multiply(kernel, a)
                    sum_res[row][col] += res.sum()

                col += 1
            row += 1
        return sum_res


cnn = CNN_Model()
img = cv2.imread('girl3.jpg', 0)/255.0
cv2.imshow('img_origin', img)
# Canh doc
kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
# Canh ngang
kernel0 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
# Lay canh
kernel1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
# Lay net
kernel2 = np.array([[0, -1, 0], [-1, 5, -1], [-0, -1, -0]])
# Lam mo
kernel3 = 1/9*np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
img = cnn.conv2d(kernel0, img)

#img = cnn.maxpooling2d((2, 2), img)
img = np.array(img)
print(img.shape)

cv2.imshow('img', img)
cv2.waitKey(0)
