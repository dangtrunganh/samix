import random
import math
from typing import Optional
import numpy as np


def pedcc_generation(
        n: int, k: int = None, seed: Optional[int] = None
) -> np.ndarray:
    def pedcc_frame(n: int, k: int = None) -> np.ndarray:
        assert 0 < k <= n + 1
        zero = [0] * (n - k + 1)
        u0 = [-1][:0] + zero + [-1][0:]
        u1 = [1][:0] + zero + [1][0:]
        u = np.stack((u0, u1)).tolist()
        for i in range(k - 2):
            c = np.insert(u[len(u) - 1], 0, 0)
            for j in range(len(u)):
                p = np.append(u[j], 0).tolist()
                s = len(u) + 1
                u[j] = math.sqrt(s * (s - 2)) / (s - 1) * np.array(p) - 1 / (
                        s - 1
                ) * np.array(c)
            u.append(c)
        return np.array(u)

    U = pedcc_frame(n=n, k=k)
    r = np.random.RandomState(seed)
    while True:
        try:
            noise = r.rand(n, n)  # [0, 1)
            V, _ = np.linalg.qr(noise)
            break
        except np.linalg.LinAlgError:
            continue
    points = np.dot(U, V)
    return points


def generate(
        d: int,
        k: int,
        path_save_prototype: str,
        seed: Optional[int] = None,
) -> np.ndarray:
    """Generate k evenly distributed R^n points in a unit (n-1)-hypersphere
    Args:
        d (int): dimension of the Euclidean space
        k (int): number of points to generate
        method (str): method to generate the points. Defaults to "simplex".
        path_save_prototype (str): filepath to save the points.
        seed (int, optional): seed for the random number generator. Defaults to None.

    Returns:
        np.ndarray: k evenly distributed points in a unit (n-1)-hypersphere
    >>> generate(2, 3, method="simplex")
    array([[ 0.        ,  0.        ],
           [ 0.70710678,  0.70710678],
    """
    assert k < d + 2
    if seed is None or not (isinstance(seed, int) and 0 <= seed < 2 ** 32):
        seed = random.randrange(2 ** 32)
    points = pedcc_generation(d, k, seed=seed)
    if path_save_prototype:
        np.save(path_save_prototype, points)
    return points
