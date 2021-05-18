"""
This file contains implementation regarding the Rappor algorithm implemented in Google
"""

from typing import Dict
import numpy as np
from numpy.core.fromnumeric import size

memorizedFilters: Dict[str, np.ndarray] = dict()


def FakeBloomFilter(B: np.ndarray, v: str, f: float):
    """Create a fake bloom filter from a given bloom filter and memorize it based on the string `v`

    Args:
        B (np.ndarray): 256-bit long bloom filter
        v (str): String to memorize
        f (float): a user-tunable parameter controlling the level of longitudinal privacy guarantee
    """

    assert 0 <= f <= 1, f"f must be between 0 and 1, current value: {f}"
    assert v not in memorizedFilters, f"v is already memorized, current fake bloom filter: {memorizedFilters[v]}"

    result = np.zeros(B.size, dtype=int)
    for i in range(result.size):
        result[i] = np.random.choice(a=[1, 0, B[i]], p=[f/2, f/2, 1-f])
    # Memorize the fake filter
    memorizedFilters[v] = result
    return result


def RandomizedResponse(B_fake: np.ndarray, q: float, p: float):
    """Generate an instantaneous randomized response from the fake bloom filter

    Args:
        B_fake (np.ndarray): Fake bloom filter memorized by the client
        q (float): Probability to set the bit to 1 if B_fake is 1 there
        p (float): Probability to set the bit to 1 if B_fake is 0 there
    """

    result = np.zeros(B_fake.size, dtype=int)
    result[B_fake == 1] = np.random.binomial(1, q, B_fake[B_fake == 1].size)
    result[B_fake == 0] = np.random.binomial(1, p, B_fake[B_fake == 0].size)
    return result


if __name__ == "__main__":
    B = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
                 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1], dtype=int)
    B_fake = FakeBloomFilter(B, "furkan", 0.05)
    C = RandomizedResponse(B_fake, 0.75, 0.50)
    print(B)
    print(B_fake)
    print(C)
