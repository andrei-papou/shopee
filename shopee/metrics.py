from typing import Tuple

import torch

from shopee.types import Triplet


def get_triplet_accuracy_components(t: Triplet, margin: float) -> Tuple[int, int]:
    n: int = t.p.shape[0] + t.n.shape[0]
    av = torch.unsqueeze(t.a, dim=1)
    pv = torch.unsqueeze(t.p, dim=1)
    nv = torch.unsqueeze(t.n, dim=1)
    n_correct = (torch.cdist(av, pv) <= margin).int().sum().item() + \
        (torch.cdist(av, nv) > margin).int().sum().item()
    return n_correct, n
