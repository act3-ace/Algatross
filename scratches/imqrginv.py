# Imports
from timeit import timeit

import numpy as np

from scipy import linalg as sla


# Function definition
def imqrginv_fixed_scipy(a: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    q, r, p = sla.qr(a, mode="economic", pivoting=True)

    r_take = np.any(np.abs(r) > tol, axis=1)
    r = r[r_take, ::]
    q = q[::, r_take]

    return np.linalg.multi_dot((r.T, np.linalg.inv(r @ r.T), q.T))[np.argsort(p), ::]


def imqrginv_fixed_numpy(a: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    q, r = np.linalg.qr(a, mode="reduced")

    r_take = np.any(np.abs(r) > tol, axis=1)
    r = r[r_take, ::]
    q = q[::, r_take]

    # return r.T @ np.linalg.solve(a=r @ r.T, b=q.T)
    return r.T @ np.linalg.inv(r @ r.T) @ q.T

    # return r.T @ sla.solve(
    #     a=r @ r.T,
    #     b=q.T,
    #     assume_a="pos",
    #     check_finite=False,
    #     overwrite_a=True,
    #     overwrite_b=True,
    # )
    # return np.linalg.multi_dot((r.T, np.linalg.inv(r @ r.T), q.T))


# Tests
# a function to generate random singular matrices
def gen_singular_matrix(n: int, m: int) -> np.ndarray:
    a = np.random.rand(n, m)
    # the matrix is made singular by setting the last row/column to be the sum of the
    # previous rows/columns
    if n > m:
        a[::, -1] = np.sum(a[::, :-1], axis=1)
    else:
        a[-1, ::] = np.sum(a[:-1, ::], axis=0)

    return a


# a test for the QR-based pseudoinverse
np.random.seed(42)
number = 1000
iters = []
for n in [5, 10, 100, 1000, 10000, 100000]:
    for m in [5, 10, 20, 30, 40, 50]:
        a = gen_singular_matrix(n, m)
        print(
            f"Log10 of Condition number: {np.log10(np.linalg.cond(a)):.4f} (n={n}, m={m})",
        )  # 🥴 roughly >= 16 for all, i.e. very ill-conditioned
        pinv_from_qr = imqrginv_fixed_numpy(a)
        imqrginv_time = 1000 * timeit(lambda: imqrginv_fixed_numpy(a), number=number) / number
        pinv_from_svd = np.linalg.pinv(a)
        np_time = 1000 * timeit(lambda: np.linalg.pinv(a), number=number) / number
        iters.append((np_time - imqrginv_time) / np_time)
        print(
            f"IMQRGINV Time (ms): {imqrginv_time:.4f} | NumPy Time (ms): {np_time:.4f} | Advantage : {100 * (np_time - imqrginv_time) / np_time:.4f}%",
        )
        print(f"Mean: {100 * np.mean(iters):0.4f}")
        print()
        assert np.allclose(pinv_from_qr, pinv_from_svd), f"Failed for n={n}, m={m}"
        # ✅ all pass
print(np.mean(iters))
