from enum import Enum
from typing import Optional


class ValueChangeDirection(str, Enum):
    up = 'up'
    down = 'down'


class ThresholdFinder:

    def __init__(
            self,
            threshold_min_value: float,
            threshold_max_value: float,
            init_mean_num_pos: float,
            min_mean_num_pos: float,
            max_mean_num_pos: float,
            max_num_iterations: int):
        assert threshold_min_value < threshold_max_value
        assert min_mean_num_pos < max_mean_num_pos
        self._threshold = (threshold_min_value + threshold_max_value) / 2
        self._threshold_min_value = threshold_min_value
        self._threshold_max_value = threshold_max_value
        self._min_mean_num_pos = min_mean_num_pos
        self._max_mean_num_pos = max_mean_num_pos
        self._mean_num_pos = init_mean_num_pos
        self._done_num_iterations = 0
        self._max_num_iterations = max_num_iterations

    @property
    def threshold(self) -> float:
        return self._threshold

    def get_next_threshold(self, mean_num_pos: float) -> Optional[float]:
        if self._min_mean_num_pos <= mean_num_pos <= self._max_mean_num_pos:
            return None
        if self._done_num_iterations >= self._max_num_iterations:
            return None
        if mean_num_pos > self._max_mean_num_pos:
            self._threshold_max_value = self._threshold
            self._threshold = (self._threshold_min_value + self._threshold) / 2
        else:
            self._threshold_min_value = self._threshold
            self._threshold = (self._threshold + self._threshold_max_value) / 2
        self._done_num_iterations += 1
        return self._threshold
