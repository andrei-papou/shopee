from typing import Dict, Any, TypedDict

import torch


class Checkpoint(TypedDict):
    model_state_dict: Dict[str, Any]
    opt_state_dict: Dict[str, Any]
    lr_scheduler: object


def save_checkpoint(checkpoint: Checkpoint, path: str):
    torch.save(checkpoint, path)


def load_checkpoint(path: str) -> Checkpoint:
    return torch.load(path)
