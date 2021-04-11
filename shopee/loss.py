import torch


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin: float):
        super().__init__()
        self._margin = margin

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param x1: Tensor of shape (N, D) where N - batch size, D - embedding dimensionality.
        Embeddings of the first batch.
        :param x2: Tensor of shape (N, D) where N - batch size, D - embedding dimensionality.
        Embeddings of the second batch.
        :param y: Tensor of shape (N,) where N - batch size, with values either 0 or 1.
        0 stands for the samples of the same class, 1 - different classes.
        :return: Tensor of shape (N,) where each element is the contrastive loss value between
        corresponding samples from x1 and x2.
        """
        dist = torch.cdist(x1.unsqueeze(1), x2.unsqueeze(1)).squeeze()
        same_mask, diff_mask = y, (~y.bool()).int()  # type: (torch.Tensor, torch.Tensor)

        loss = dist * same_mask + torch.max(
            torch.ones(*dist.shape).cuda() * self._margin - dist,
            torch.zeros(*dist.shape).cuda()) * diff_mask
        return loss.mean()
