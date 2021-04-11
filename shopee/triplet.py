import torch


def get_label_similarity_matrix(labels: torch.Tensor) -> torch.Tensor:
    """
    :param labels: Tensor of shape (N,) where N is the dataset size. n-th element of the tensor is the label
    of the n-th element of the dataset.
    :return: Tensor of shape (N, N). (i, j)-th element is 1 if i-th and j-th samples belong to the same label,
    otherwise 0.
    """

    rows = torch.tile(torch.unsqueeze(labels, dim=1), dims=(1, len(labels)))
    cols = torch.tile(torch.unsqueeze(labels, dim=0), dims=(len(labels), 1))
    return torch.eq(rows, cols).int()


def get_hardest_pos_neg(embed_tensor: torch.Tensor, label_similarity_matrix: torch.Tensor) -> torch.Tensor:
    """
    :param embed_tensor: Tensor of shape (N, D) where N is the dataset size and D is the dimension of the embedding.
    It contains embedding vector for each sample.
    :param label_similarity_matrix: Tensor of shape (N, N), see `get_label_similarity_matrix` docs for explanation.
    :return: Tensor of shape (N, 2) where (n, 0) is the index of the hardest positive sample for the n-th sample,
    (n, 1) is the index of the hardest negative sample for the n-th sample.
    """
    n, _ = embed_tensor.shape
    norm_sq = torch.tile(torch.sum(embed_tensor ** 2, dim=1, keepdim=True), dims=(1, n))
    dot_prod = torch.dot(embed_tensor, embed_tensor.T)
    dist_matrix = norm_sq - 2 * dot_prod + norm_sq.T

    hardest_pos = (dist_matrix * label_similarity_matrix).argmax(dim=1, keepdim=True)
    neg_masked = dist_matrix * label_similarity_matrix.logical_not().int()
    hardest_neg = (neg_masked + (neg_masked <= 0).int() * torch.finfo(torch.float32).max).argmin(dim=1, keepdim=True)

    return torch.cat([hardest_pos, hardest_neg], dim=1)
