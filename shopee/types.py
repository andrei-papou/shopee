from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from typing_extensions import TypedDict

import torch


class OptimizerConfig(TypedDict):
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau
    monitor: str


TripletTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class Triplet:
    a: torch.Tensor
    p: torch.Tensor
    n: torch.Tensor

    @classmethod
    def from_first_dim_split(cls, x: torch.Tensor) -> Triplet:
        num_samples = int(x.shape[0])
        split_size = num_samples // 3
        return cls(a=x[:split_size], p=x[split_size:split_size * 2], n=x[split_size * 2:])

    @classmethod
    def from_tuple(cls, t: TripletTuple) -> Triplet:
        return cls(a=t[0], p=t[1], n=t[2])

    def join(self, dim: int = 0) -> torch.Tensor:
        return torch.cat([self.a, self.p, self.n], dim=dim)
