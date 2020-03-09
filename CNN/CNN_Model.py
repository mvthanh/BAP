import numpy as np


class CNN_Model:
    def __init__(self):
        self.model = []
        self.data = np.array([])

    def flatten(self):
        new_arr = np.array(self.data)
        return np.reshape(new_arr, (-1))

    def maxpooling2d(self, pool_size):
        input_shape = self.data.shape
        l = len(input_shape)
        if len(pool_size) != 2:
            print("Filter or input size is not valid")
            exit(1)
        if pool_size[0] > input_shape[0] or pool_size[1] > input_shape[1]:
            print("Filter is not valid")
            exit(1)
        if l == 2:
            input_shape = (input_shape[0], input_shape[1], 1)
        result = []
        for layer in range(input_shape[2]):
            col = 0
            res_col = []
            while col < input_shape[1]:
                row = 0
                res_row = []
                while row < input_shape[0]:
                    a = self.data[row: min(row + pool_size[0], input_shape[0]), col: min(col + pool_size[1], input_shape[1])][layer]
                    res_row.append(np.max(a))
                    row += pool_size[0]
                res_col.append(res_row)
                col += pool_size[1]
            result.append(res_col)

    def conv2d(self, filters, filter_size, input_shape):
        l = len(input_shape)
        if l != 2 or l != 3 or len(filter_size) != 2:
            print("Filter or input size is not valid")
            exit(1)
        if filter_size[0] > input_shape[0] or filter_size[1] > input_shape[1]:
            print("Filter is not valid")
            exit(1)
        if l == 2:
            input_shape = (input_shape[0], input_shape[1], 1)
        result = []
        for filter in range(filters):
            sum_res = np.zeros((input_shape[0] - filters[0] + 1, input_shape[1] - filters[1] + 1))
            for id in range(input_shape[2]):
                kernel = np.random.randint(-1, 2, filter_size[0] * filter_size[1])
                col = 0
                while (col + filter_size[1]) < input_shape[1]:
                    row = 0
                    while (row + filter_size[0]) < input_shape[0]:
                        if l == 2:
                            a = self.data[row: row + filter_size[0], col: col + filter_size[1]]
                        else:
                            a = self.data[row: row + filter_size[0], col: col + filter_size[1]][id]
                        a = np.reshape(a, (-1))
                        res = np.multiply(kernel, a)
                        sum_res[row][col] += res
                        row += 1
                    col += 1
            result.append(sum_res)
