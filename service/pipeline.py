from pathlib import Path
from typing import Set, List

import numpy as np
import pipeline_functions
import redis
import pickle

import settings


class Pipeline:
    def __init__(self, path: Path, warmup_duration: int, name: str = "untitled"):
        self._states: dict = {}
        self._pipeline: list = []
        self._names: Set[str] = set()
        self._path: Path = path

        self._name = name
        self._redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB_NUM)

        self.warmup_duration = warmup_duration
        self._redis_client.set(self._name + "~init", pickle.dumps([]))
        self._redis_client.set(self._name + "~init_len", 0)

        self._processing_function = self._init

    def set_name(self, name: str) -> str:
        i = 0
        while True:
            new_name = f'{self._name}_{name}_{i}'
            if new_name not in self._names:
                self._names.add(new_name)
                break
            i += 1
        return new_name

    def get_redis_client(self) -> redis.Redis:
        return self._redis_client

    def reset(self):
        pass

    def _build_pipeline(self, init_x: np.ndarray):
        self._pipeline = []
        with open(self._path) as f:
            x = init_x
            for line in f:
                pipeline_function: pipeline_functions.PipelineFunction = eval("pipeline_functions." + line)
                pipeline_function.set_parent(self)
                pipeline_function.set_initial_state(x)
                x = pipeline_function.compute(x)
                self._pipeline.append(pipeline_function)

        return x

    def process(self, x: np.ndarray) -> np.ndarray:  # Generator state machine
        return self._processing_function(x)

    def _process(self, x: np.ndarray) -> np.ndarray:
        for pipeline_function in self._pipeline:
            x = pipeline_function.compute(x)
        return x

    def _init(self, x: np.ndarray) -> np.ndarray:
        # Get the latest warmup x
        init: List[bytes] = pickle.loads(self._redis_client.get(self._name + "~init"))
        init_len: int = int(self._redis_client.get(self._name + "~init_len"))

        # Update state
        init.append(x.dumps())
        init_len += len(x)

        # Case if not enough data to perform inference
        if init_len < self.warmup_duration:
            raise PipelineHasNotBeenInitializedException

        # When there are enough data, end initialization state
        init_np: List[np.ndarray] = [pickle.loads(b) for b in init]
        x = np.concatenate(init_np, axis=0)
        x = self._build_pipeline(x)
        self._processing_function = self._process

        return x

        # Save state
        self._redis_client.set(self._name + "~init", pickle.dumps(init))
        self._redis_client.set(self._name + "~init_len", init_len)

        raise


class PipelineHasNotBeenInitializedException(Exception):
    pass