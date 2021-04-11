from __future__ import annotations

from typing import TypeVar, Iterable, Generic

T = TypeVar('T')


class NIter(Generic[T]):

    def __init__(self, iterable: Iterable[T], n: int):
        self._iterator = iter(iterable)
        self._n_left = n

    def __next__(self) -> T:
        if self._n_left > 0:
            self._n_left -= 1
            return next(self._iterator)
        raise StopIteration()

    def __iter__(self) -> NIter[T]:
        return self
