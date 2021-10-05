from math import sqrt
from typing import Tuple

import numpy as np
from scipy.stats import rankdata


def math_round(x: float, digit: int = 0) -> int:
    # In python there is no MATH round in std lib, so here's a hack
    return int(np.round(x, digit))


def calc_rank(arr: np.ndarray) -> Tuple[int, int, float]:
    arr = arr[np.argsort(arr[:, 0])]

    ranks = rankdata(arr[:, 1])
    # Now rank is from 1 to N
    # We need from N to 1
    ranks -= ranks.max() + 1
    ranks *= -1

    n = arr.shape[0]
    p = math_round(n / 3)

    r1 = ranks[:p].sum()
    r2 = ranks[-p:].sum()
    diff = r1 - r2
    error = math_round((n + 0.5) * sqrt(p / 6))
    corr = diff / (p * (n - p))

    return math_round(diff), error, np.round(corr, 2)


if __name__ == '__main__':
    def main():
        arr = np.loadtxt("in.txt", delimiter=" ")
        assert arr.ndim == 2 and arr.shape[1] == 2
        if arr.shape[0] < 9:
            print(f"There's just {arr.shape[0]} elements which is less than 9 and isn't enough for calculation")

        # Sort by x
        diff, error, corr = calc_rank(arr)

        with open("out.txt", "w") as f:
            f.write(f"{diff} {error} {corr}\n")

    main()
