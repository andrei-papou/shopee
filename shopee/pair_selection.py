import torch
import torch.nn.functional as torch_f

from shopee.types import Triplet


def get_label_similarity_matrix(labels: torch.Tensor) -> torch.Tensor:
    rows = torch.tile(torch.unsqueeze(labels, dim=1), dims=(1, len(labels)))
    cols = torch.tile(torch.unsqueeze(labels, dim=0), dims=(len(labels), 1))
    return torch.eq(rows, cols).int()


def get_hardest_triplet(
        embed_tensor: torch.Tensor,
        label_similarity_matrix: torch.Tensor,
        distance: str) -> Triplet:
    n, _ = embed_tensor.shape
    if distance == 'euclidean':
        norm_sq = torch.tile(torch.sum(embed_tensor ** 2, dim=1, keepdim=True), dims=(1, n))
        dot_prod = torch.matmul(embed_tensor, embed_tensor.T)
        dist_matrix = norm_sq - 2 * dot_prod + norm_sq.T
    elif distance == 'cosine':
        n, _ = embed_tensor.shape
        embed_tensor = torch_f.normalize(embed_tensor, dim=1)
        dist_matrix = 1.0 - torch.matmul(embed_tensor, embed_tensor.T)
    else:
        raise NotImplementedError(f'Unsupported distance: {distance}.')

    hardest_pos = (dist_matrix * label_similarity_matrix).argmax(dim=1, keepdim=True)
    neg_masked = dist_matrix * label_similarity_matrix.logical_not().int()
    hardest_neg = (neg_masked + (neg_masked <= 0).int() * torch.finfo(torch.float32).max).argmin(dim=1, keepdim=True)

    return Triplet(
        a=embed_tensor,
        p=embed_tensor[hardest_pos, :].squeeze(dim=1),
        n=embed_tensor[hardest_neg, :].squeeze(dim=1))


