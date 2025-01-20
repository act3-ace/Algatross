import gc
import multiprocessing

from concurrent.futures import ProcessPoolExecutor
from timeit import timeit

import numpy as np

import torch

import torch_tensorrt  # noqa: F401

from algatross.extreme_learning_machines.utils.ops import geninv, imqrginv

# os.environ["GPU_NUM_DEVICES"] = "1"
# os.environ["PJRT_DEVICE"] = "TPU"


def check_speed(number, algo, setup):
    result = timeit("algo(a)", setup=setup, number=number, globals={"algo": algo, "torch": torch, "np": np, "gc": gc}) / number
    return result


# print((f == n).all())

if __name__ == "__main__":

    number = 10

    # a = np.random.rand(10000, 1000)
    # # a = np.random.rand(1000, 500)
    # n = np.linalg.pinv(a)
    # f = geninv(a)
    # # cf = c_geninv(a)
    # # qf, P = imqrginv(a)
    # qf = imqrginv(a)
    # np_t = timeit(lambda: np.linalg.pinv(a), number=number) / number
    # np_fast_t = timeit(lambda: geninv(a), number=number) / number
    # # c_fast_t = timeit(lambda: c_geninv(a), number=number) / number
    # np_i_fast_t = timeit(lambda: imqrginv(a), number=number) / number
    # print((f == n).all())

    # a = torch.rand(4, 3, dtype=torch.float64)
    # a = torch.rand(10000, 1000, dtype=torch.float64, device="cuda")
    a = torch.rand(10000, 1000, dtype=torch.float32, device="cuda")
    # a = torch.rand(10000, 1000, dtype=torch.float32, device="cpu")
    n = torch.linalg.pinv(a)
    torch_t = timeit(lambda: torch.linalg.pinv(a), number=number) / number
    torch_fast_t = timeit(lambda: geninv(a), number=number) / number
    # c_fast_t = timeit(lambda: c_geninv(a), number=number) / number
    torch_i_fast_t = timeit(lambda: imqrginv(a), number=number) / number
    print("torch.linalg.pinv result: ", torch_t)
    print("torch.geninv result: ", torch_fast_t)
    print("torch.imqrginv result: ", torch_i_fast_t)
    f = geninv(a)

    futures = []
    for algo, setup in [
        # (np.linalg.pinv, "gc.enable(); a = np.random.rand(10_000, 1_000)"),
        # (geninv, "gc.enable(); a = np.random.rand(10_000, 1_000)"),
        # (imqrginv, "gc.enable(); a = np.random.rand(10_000, 1_000)"),
        # (torch.linalg.pinv, "gc.enable(); a = torch.rand(10_000, 1_000, dtype=torch.float64, device='cuda')"),
        # (geninv, "gc.enable(); a = torch.rand(10_000, 1_000, dtype=torch.float64, device='cuda')"),
        # (imqrginv, "gc.enable(); a = torch.rand(10_000, 1_000, dtype=torch.float64, device='cuda')"),
        (np.linalg.pinv, "a = np.random.rand(10_000, 1_000)"),
        (geninv, "a = np.random.rand(10_000, 1_000)"),
        (imqrginv, "a = np.random.rand(10_000, 1_000)"),
        # (torch.linalg.pinv, "a = torch.rand(10_000, 1_000, dtype=torch.float64, device='cuda')"),
        # (geninv, "a = torch.rand(10_000, 1_000, dtype=torch.float64, device='cuda')"),
        # (imqrginv, "a = torch.rand(10_000, 1_000, dtype=torch.float64, device='cuda')"),
        (torch.linalg.pinv, "a = torch.rand(10_000, 1_000, dtype=torch.float32, device='cuda')"),
        (geninv, "a = torch.rand(10_000, 1_000, dtype=torch.float32, device='cuda')"),
        (imqrginv, "a = torch.rand(10_000, 1_000, dtype=torch.float32, device='cuda')"),
        # (torch.linalg.pinv, "a = torch.rand(10_000, 1_000, dtype=torch.float32, device='cpu')"),
        # (geninv, "a = torch.rand(10_000, 1_000, dtype=torch.float32, device='cpu')"),
        # (imqrginv, "a = torch.rand(10_000, 1_000, dtype=torch.float32, device='cpu')"),
    ]:
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pex:
            futures.append(pex.submit(check_speed, number=10, algo=algo, setup=setup))
    for test, future in zip(
        ["np.linalg.pinv", "np.geninv", "np.imqrginv", "torch.linalg.pinv", "torch.geninv", "torch.imqrginv"],
        futures,
        strict=True,
    ):
        print(f"{test} result: ", future.result())
