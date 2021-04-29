import torch
import torch.nn.functional as torch_f


def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch_f.cosine_similarity(x, y)
