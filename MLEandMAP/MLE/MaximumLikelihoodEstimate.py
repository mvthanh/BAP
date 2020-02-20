import numpy as np


class MaximumLikelihoodEstimation:
    def __init__(self, type):
        self.type = type

    def pp(self):
        """
        gia su tung dong xu N lan va duoc n mat head.
        Uoc luong tham so lamda - xac suat cua mat head
        """

        def BNL(n, N):
            return n / N

        """
        Gia su tung vien xuc xac sau mat. Trong N lan tung, 
        so luong xuat hien cac mat t1, t2, ... t6 la n1, n2, ...n6.
        Uoc luong xac suat roi vao moi mat.
        """

        def Categorical(n, face_i):
            n_new = np.array([n], dtype=np.float64)
            n_new = n_new / n_new.sum()
            if face_i > len(n):
                return 'invalid'
            return n_new[0][face_i - 1]

        """
        Gia su du lieu thu duoc la cac gia tri nhieu chieu
        Danh gia cac tham so vector ky vong va ma tran hiep phuong sai 
        cua pp multivariate normal distribution
        """
        def Multivarial(t, data):
            def kyvong(data):
                data.mean(0)

            def MTHphuongsai(data):
                u = data.mean(0)
                sum = np.zeros((u.shape[0], u.shape[0]), dtype=np.float64)
                for xi in data:
                    sum = sum + np.dot((xi - u), (xi - u).T)
                return sum / data.shape[0]

            switcher1 = {
                'kv': kyvong(data),
                'ps': MTHphuongsai(data)
            }
            return switcher1.get(t, 'Invalid')

        """
        Khi thuc hien mot phep do, gia su rat kho de do chinh xac
        do dai cua 1 vat. Ngta do nhieu lan roi suy ra kq. 
        Voi gia thiet cac lan do doc lap.
        Uoc luong cac tham so ky vong, psai cua mo hinh nay theo pp Univariate normaldistribution
        """
        def Univariate(t, data):
            def kyvong(data):
                new = np.array([data])
                return new.mean()

            def phuongsai(data):
                new = np.array([data])
                new = (new - new.mean())**2
                return new.sum()/new.shape[1]

            switcher2 = {
                'kv': kyvong(data),
                'ps': phuongsai(data)
            }
            return switcher2.get(t, 'Invalid')

        switcher = {
            'BNL': BNL,
            'Categorical': Categorical,
            'Multivarial': Multivarial,
            'Univariate': Univariate
        }
        return switcher.get(self.type, 'Invalid')


mle = MaximumLikelihoodEstimation('Categorical')
k = mle.pp()([1, 1, 1, 1], 4)
print(k)

MLE = MaximumLikelihoodEstimation('Univariate')
k = MLE.pp()('ps', [1, 2, 3, 4])
k1 = MLE.pp()('kv', [1, 2, 3, 4])
print(k, k1)
