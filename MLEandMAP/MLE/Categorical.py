"""
Gia su tung vien xuc xac sau mat. Trong N lan tung, 
so luong xuat hien cac mat t1, t2, ... t6 la n1, n2, ...n6.
Uoc luong xac suat roi vao moi mat.
"""


class Categorical:
    def __init__(self, n=[1, 2, 1, 1, 1, 1]):
        self.n = n

    def lamda(self, iface):
        N = 0
        for i in self.n:
            N += i
        return 1.0 * self.n[iface - 1] / N


cat = Categorical()
print(cat.lamda(2))
