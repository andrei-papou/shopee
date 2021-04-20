import random
from typing import List

import pandas as pd
from torch.utils.data import Sampler


class _TripletOnlineIter:

    def __init__(self, df: pd.DataFrame, candidate_label_list: List[int], batch_size: int, num_batches: int):
        self._df = df
        self._batch_size = batch_size
        self._num_batches_total = num_batches
        self._num_batches_delivered = 0
        self._candidate_label_list = candidate_label_list
        self._queue: List[int] = []

    def _enqueue_label_group(self, lg: int, n: int):
        self._queue.extend(self._df[self._df.label_group == lg].sample(n=n).index.to_list())

    def __next__(self) -> int:
        if self._queue:
            return self._queue.pop()
        self._num_batches_delivered += 1
        if self._num_batches_delivered >= self._num_batches_total:
            raise StopIteration()
        lg1, lg2 = random.sample(self._candidate_label_list, 2)
        self._enqueue_label_group(lg=lg1, n=self._batch_size // 2)
        self._enqueue_label_group(lg=lg2, n=self._batch_size - self._batch_size // 2)
        return self._queue.pop()


class TripletOnlineSampler(Sampler[int]):

    def __init__(self, df: pd.DataFrame, batch_size: int, num_batches: int):
        super().__init__(None)
        self._df = df
        self._batch_size = batch_size
        self._num_batches = num_batches
        lg_counts = df.groupby('label_group').posting_id.count()
        self._candidate_label_list: List[int] = lg_counts[lg_counts >= batch_size // 2].index.tolist()

    def __len__(self) -> int:
        return self._num_batches * self._batch_size

    def __iter__(self) -> _TripletOnlineIter:
        return _TripletOnlineIter(
            df=self._df,
            batch_size=self._batch_size,
            num_batches=self._num_batches,
            candidate_label_list=self._candidate_label_list)
