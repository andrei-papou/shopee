import abc
from typing import List, Type

import torch
from timm import create_model
from timm.models.efficientnet import EfficientNet as _EfficientNet
from timm.models.resnet import ResNet as _ResNet


class Backbone(torch.nn.Module, metaclass=abc.ABCMeta):

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().__call__(x, *args, **kwargs)

    @property
    @abc.abstractmethod
    def num_features(self) -> int:
        ...

    @classmethod
    @abc.abstractmethod
    def get_label_prefix(cls) -> str:
        ...


class ResNet(Backbone):

    def __init__(self, pretrained: bool = True, version: str = '34'):
        super().__init__()
        self._model: _ResNet = create_model(model_name=f'resnet{version}', pretrained=pretrained)
        self._model.global_pool = torch.nn.Identity()
        self._model.fc = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model.forward_features(x)

    @property
    def num_features(self) -> int:
        return self._model.num_features

    @classmethod
    def get_label_prefix(cls) -> str:
        return 'resnet'


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

    @classmethod
    def get_label_prefix(cls) -> str:
        return 'efficientnet'


def create_backbone(label: str, pretrained: bool = True) -> Backbone:
    if label.startswith(ResNet.get_label_prefix()):
        return ResNet(pretrained=pretrained, version=label.replace(ResNet.get_label_prefix(), ''))
    elif label.startswith(EfficientNet.get_label_prefix()):
        return EfficientNet(pretrained=pretrained, version=label.replace(EfficientNet.get_label_prefix(), ''))
    raise ValueError(f'Unsupported backbone: {label}.')
