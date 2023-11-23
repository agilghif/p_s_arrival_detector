from collections import deque
from typing import Iterable, List, Tuple

import numpy as np


class StreamMaxFinder:
    def __init__(self, init_window: np.ndarray):
        channels: int = init_window.shape[-1]
        self.deque_list: List[deque] = [None for _ in range(channels)]
        self.lifetimes: List[int] = [0 for _ in range(channels)]
        self.max_vals: List = [0 for _ in range(channels)]
        self.channels: int = channels
        self.lifetime: int = init_window.shape[0] - 1

        for c in range(channels):
            window_c = init_window[:,c]
            argmax = window_c.argmax()
            max_val = window_c[argmax]

            self.deque_list[c] = deque(window_c, init_window.shape[0])
            self.lifetimes[c] = argmax
            self.max_vals[c] = max_val

    def insert(self, val: np.ndarray) -> np.ndarray:
        for c in range(self.channels):
            new_val = val[c]

            # Delete leftmost element
            self.deque_list[c].popleft()
            self.deque_list[c].append(new_val)
            self.lifetimes[c] -= 1

            # Case maximum element is already out of window
            if self.lifetimes[c] < 0:
                new_max, new_lifetime = StreamMaxFinder._deque_max(self.deque_list[c])
                self.max_vals[c] = new_max
                self.lifetimes[c] = new_lifetime

            # Case if new values is greater (or equal) than current maximum value
            elif new_val >= self.max_vals[c]:
                self.lifetimes[c] = self.lifetime
                self.max_vals[c] = new_val

        return np.array(self.max_vals)

    @staticmethod
    def _deque_max(d: deque) -> Tuple[np.float_,int]:
        """return the max and argmax value in deque"""
        curr_max = d[0]
        argmax = 0
        for idx, val in enumerate(d):
            if val >= curr_max:
                curr_max = val
                argmax = idx
        return curr_max, argmax


if __name__ == '__main__':
    from random import randint
    from time import monotonic

    # Sanity check
    # Set window size = 5
    init_window = np.array([[randint(0,100),randint(0,100)] for i in range(5)])
    a = np.array([[randint(0,100),randint(0,100)] for i in range(3,9)])

    print("initial window:")
    print(init_window)
    print("\ncontinuation:")
    print(a)

    msf = StreamMaxFinder(init_window)
    print("\nmoving windows result:")
    print(msf.max_vals)

    for v in a:
        print(msf.insert(v))

    # Speed test
    WINDOW_SIZE = 300
    init_window = np.array([[randint(0,10000) for j in range(WINDOW_SIZE)] for i in range(100)])
    a = np.array([[randint(0,10000) for j in range(WINDOW_SIZE)] for i in range(10000)])

    # Manual Max Finding
    concatenated = np.concatenate([init_window, a])
    start = monotonic()
    for i in range(len(concatenated)-WINDOW_SIZE):
        np.max(concatenated[i:i+WINDOW_SIZE], axis=0)
    end = monotonic()
    print(f"Manual max finding: {end-start} ns")

    # Stream Max Finder
    msf = StreamMaxFinder(init_window)
    start = monotonic()
    for val in a:
        msf.insert(val)
    end = monotonic()
    print(f"Stream max finding: {end - start} s")
