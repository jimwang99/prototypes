import torch
import numpy as np

from perf import PerfMonitor
from tqdm import tqdm
from loguru import logger

from typing import Tuple

import lib


def _test_matmul(
    func,
    dtype: np.dtype,
    n_row: int,
    n_col: int,
    n_inner: int,
    num_iter: int = 1000,
) -> None:
    logger.debug(f"{dtype=} {n_row=} {n_col=} {n_inner=} {num_iter=}")
    assert dtype in [np.int32, np.float32]

    pmon_np = PerfMonitor(
        logger,
        f"numpy_{dtype}_{n_row}_{n_col}_{n_inner}",
        print_iterations=num_iter,
        unit="us",
    )

    pmon_torch = PerfMonitor(
        logger,
        f"torch_{dtype}_{n_row}_{n_col}_{n_inner}",
        print_iterations=num_iter,
        unit="us",
    )

    pmon_lib = PerfMonitor(
        logger,
        f"lib_{dtype}_{n_row}_{n_col}_{n_inner}",
        print_iterations=num_iter,
        unit="us",
    )


    for _ in tqdm(range(num_iter)):
        if dtype == np.float32:
            a = np.random.randn(n_row, n_inner)
            b = np.random.randn(n_inner, n_col)
        elif dtype == np.int32:
            a = np.random.randint(low=0, high=256, size=(n_row, n_inner), dtype=np.int32)  # type: ignore
            b = np.random.randint(low=0, high=256, size=(n_inner, n_col), dtype=np.int32)  # type: ignore
        else:
            raise NotImplementedError(f"{dtype=}")

        pmon_np.begin()
        z = np.matmul(a, b, dtype=dtype)
        pmon_np.end()
        assert z.shape == (n_row, n_col), z.shape

        _a = torch.from_numpy(a)
        _b = torch.from_numpy(b)
        pmon_torch.begin()
        _z = torch.matmul(_a, _b)
        pmon_torch.end()
        assert _z.shape == (n_row, n_col), _z.shape

        pmon_lib.begin()
        r = func(a, b)
        pmon_lib.end()
        assert np.array_equiv(r, z), f"{r.shape=} {z.shape=} {r=} {z=}"


def test_matmul_int32_small():
    _test_matmul(lib.matmul_int32, np.int32, 2, 3, 4)

def test_matmul_int32_medium():
    _test_matmul(lib.matmul_int32, np.int32, 20, 30, 40)

def test_matmul_int32_large():
    _test_matmul(lib.matmul_int32, np.int32, 200, 300, 400)

def test_matmul_int32_huge():
    _test_matmul(lib.matmul_int32, np.int32, 2000, 3000, 4000)


def test_matmul_float32_small():
    _test_matmul(lib.matmul_float32, np.float32, 2, 3, 4)

def test_matmul_float32_medium():
    _test_matmul(lib.matmul_float32, np.float32, 20, 30, 40)

def test_matmul_float32_large():
    _test_matmul(lib.matmul_float32, np.float32, 200, 300, 400)

def test_matmul_float32_huge():
    _test_matmul(lib.matmul_float32, np.float32, 2000, 3000, 4000)

if __name__ == "__main__":
    test_matmul_int32_large()
    test_matmul_float32_large()