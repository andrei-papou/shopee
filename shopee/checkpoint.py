from typing import Dict, Any, OrderedDict

import torch
from typing_extensions import TypedDict


class Checkpoint(TypedDict):
    model_state_dict: OrderedDict[str, torch.Tensor]
    opt_state_dict: Dict[str, Any]
    lr_scheduler_state_dict: Dict[str, Any]


def save_checkpoint(checkpoint: Checkpoint, path: str):
    torch.save(checkpoint, path)


def load_checkpoint(path: str) -> Checkpoint:
    return torch.load(path)
