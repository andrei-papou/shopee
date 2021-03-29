import torch
from torchvision.models import resnet18


class ResNet18(torch.nn.Module):

    def __init__(self):
        super().__init__()
        model = resnet18(pretrained=True)
        model.fc = torch.nn.Identity()
        self._model = model

    def freeze_all_but_last(self):
        for p in self._model.parameters():
            p.requires_grad_(False)
        for p in self._model.fc.parameters():
            p.requires_grad_(True)

    def unfreeze_all(self):
        for p in self._model.parameters():
            p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)
