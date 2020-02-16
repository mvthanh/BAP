"""
gia su tung dong xu N lan va duoc n mat head.
Uoc luong tham so lamda - xac suat cua mat head 
"""


class Bernoulli:
    def __init__(self, n=1, N=2):
        self.n = n
        self.N = N

    def lamda(self):
        return 1.0 * self.n / self.N


ber = Bernoulli(7, 9)
print(ber.lamda())
