import abc

import torch
from timm import create_model
from timm.models.efficientnet import EfficientNet as _EfficientNet
from torchvision.models import resnet18


class Backbone(torch.nn.Module, metaclass=abc.ABCMeta):

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().__call__(x, *args, **kwargs)

    @property
    @abc.abstractmethod
    def num_features(self) -> int:
        ...


class ResNet18(Backbone):

    def __init__(self, pretrained: bool = True):
        super().__init__()
        model = resnet18(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = torch.nn.Identity()
        self._model = model
        self._num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    @property
    def num_features(self) -> int:
        return self._num_features


class EfficientNet(Backbone):

    def __init__(self, pretrained: bool = True, version: str = 'b3'):
        super().__init__()
        self._model: _EfficientNet = create_model(model_name=f'efficientnet_{version}', pretrained=pretrained)
        self._model.classifier = torch.nn.Identity()
        self._model.global_pool = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model.forward_features(x)

    @property
    def num_features(self) -> int:
        return self._model.num_features
