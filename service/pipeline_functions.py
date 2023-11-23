from abc import ABC, abstractmethod
from typing import List

import numpy as np
import redis
import pickle


class PipelineFunction(ABC):
    def __init__(self):
        self._parent = None  # is a Pipeline instance from pipeline.py, typing is omitted due to error in bentoML
        self._redis_client: redis.Redis = None
        self._func_name: str = ""
        self._name: str = ""

    def set_parent(self, parent):
        self._parent = parent
        self._redis_client = parent.get_redis_client()
        self._name = parent.set_name(self._func_name)

    @abstractmethod
    def compute(self, x: np.ndarray) -> np.ndarray:
        pass

    def set_initial_state(self, x: np.ndarray) -> None:
        pass

    def set(self, x: np.ndarray):
        self._redis_client.set(self._name, x.dumps())

    def get(self) -> np.ndarray:
        return pickle.loads(self._redis_client.get(self._name))


class Slope(PipelineFunction):
    def __init__(self, sampling_rate: float):
        super().__init__()
        self._func_name = "slope"

        self.sampling_rate = sampling_rate

    def compute(self, x: np.ndarray) -> np.ndarray:
        prev_val = self.get()
        first_slope = (x[0] - prev_val) * self.sampling_rate
        rest_slope = (x[1:] - x[:-1]) * self.sampling_rate
        self.set(rest_slope[-1:])
        return np.concatenate([first_slope, rest_slope], axis=0)

    def set_initial_state(self, x: np.ndarray) -> None:
        self.set(x[:1])


class ExponentialSmoothing(PipelineFunction):
    def __init__(self, alpha: float, avg_until: int = 0):
        super().__init__()
        self._func_name = "ema"

        assert 0.0 <= alpha <= 1.0, "Alpha must be in range [0.0, 1.0]"

        self.alpha = alpha
        self.alpha_ = 1 - self.alpha
        self.avg_until = avg_until

    def compute(self, x: np.ndarray) -> np.ndarray:
        ema_vals: List[np.ndarray] = []
        prev_ema_val = self.get()
        for val in x:
            curr_ema_val = prev_ema_val * self.alpha + val * self.alpha_
            prev_ema_val = curr_ema_val
            ema_vals.append(curr_ema_val)
        self.set(ema_vals[-1])
        return np.array(ema_vals)

    def set_initial_state(self, x: np.ndarray) -> None:
        if self.avg_until > 0:
            self.set(np.mean(x[:self.avg_until], axis=0))
        else:
            self.set(np.zeros(x.shape[-1]))


class MultiExponentialSmoothing(PipelineFunction):
    def __init__(self, alpha: List[float], avg_until: int = 0):
        super().__init__()
        self._func_name = "mema"

        for a in alpha:
            assert 0.0 <= a <= 1.0, "Alpha must be in range [0.0, 1.0]"

        self.alpha = np.array(alpha)
        self.alpha_ = 1 - self.alpha
        self.avg_until = avg_until

    def compute(self, x: np.ndarray) -> np.ndarray:
        ema_vals: List[np.ndarray] = []
        prev_ema_val = self.get()
        for val in x:
            curr_ema_val = prev_ema_val * self.alpha + val * self.alpha_
            prev_ema_val = curr_ema_val
            ema_vals.append(curr_ema_val)
        self.set(ema_vals[-1])
        return np.array(ema_vals)

    def set_initial_state(self, x: np.ndarray) -> None:
        if self.avg_until > 0:
            self.set(np.mean(x[:self.avg_until], axis=0))
        else:
            self.set(np.zeros(self.alpha.shape[0]))


class Square(PipelineFunction):
    def __init__(self):
        super().__init__()

    def compute(self, x: np.ndarray) -> np.ndarray:
        return x*x


class Log1P(PipelineFunction):
    def __init__(self):
        super().__init__()

    def compute(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(x)


class AddChannels(PipelineFunction):
    def __init__(self):
        super().__init__()

    def compute(self, x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=-1, keepdims=True)


class PairwiseRatio(PipelineFunction):
    def __init__(self, alpha: List[float]):
        super().__init__()

        # Generate Order
        idx_area = [(i, 1-a) for i, a in enumerate(alpha)]
        triplets = [(j, i, idx_area[i][1] / idx_area[j][1]) for i in range(len(alpha)) for j in range(len(alpha)) if
                    i > j]
        triplets = sorted(triplets, key=lambda t: t[2])
        self.order = [(t[0], t[1]) for t in triplets]


    def compute(self, x: np.ndarray) -> np.ndarray:
        new_x = []
        for i, j in self.order:
            new_x.append((x[:,i]+1)/(x[:,j]+1))
        return np.array(new_x).T


class SlidingWindow(PipelineFunction):
    def __init__(self, window_size: int, normalize_windows: bool =True):
        super().__init__()
        self._func_name = "sliwin"

        self.window_size: int = window_size
        self.normalize_windows: bool = normalize_windows

    def compute(self, x: np.ndarray) -> np.ndarray:

        window_size_longer: bool = self.window_size >= len(x)
        uncut_windows = []
        cut_windows = []

        # Uncut windows
        for i in range(self.window_size, len(x)):
            uncut_windows.append(x[i-self.window_size:i])
        uncut_windows = np.array(uncut_windows)

        # Cut windows
        prev_window = self.get()
        for i in range(1, min(self.window_size, len(x))):
            cut_windows.append(np.concatenate([prev_window[i:], x[:i]]))
        cut_windows = np.array(cut_windows)

        # Update last windows
        windows: np.ndarray
        if uncut_windows.any():  # if uncut windows is not empty
            windows = np.concatenate([cut_windows, uncut_windows], axis=0)
        else:
            windows = cut_windows
        self.set(windows[-1])

        if self.normalize_windows:
            for i in range(len(windows)):
                windows_i = windows[i]
                windows_max = np.max(windows_i, axis=0)
                windows_min = np.min(windows_i, axis=0)

                windows[i] = (windows_i - windows_min) / (windows_max - windows_min)

        return windows

    def set_initial_state(self, x: np.ndarray) -> None:
        self.set(x[:self.window_size])