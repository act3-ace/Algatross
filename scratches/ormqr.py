from timeit import timeit

import torch

a = torch.rand(10000, 1000)
number = 10


def qrf(a):
    h, tau = torch.geqrf(a)
    m = min(*h.shape[-2:])
    h = torch.ormqr(h, tau, torch.eye(*a.shape))
    # m, n = h.shape[-2:]
    return h, h.triu()[:m]


def qr(a):
    Q, R = torch.linalg.qr(a, mode="reduced")
    return Q, R


qrf_t = timeit(lambda: qrf(a), number=number) / number
qr_t = timeit(lambda: qr(a), number=number) / number

print(f"QRF time: {qrf_t}")
print(f"QR time: {qr_t}")
