import abc
from typing import Union, List

import torch
from timm import create_model
from timm.models.efficientnet import EfficientNet as _EfficientNet
from timm.models.resnet import ResNet as _ResNet
from timm.models.swin_transformer import SwinTransformer as _SwinTransformer


class Backbone(torch.nn.Module, metaclass=abc.ABCMeta):

    def __call__(self, x: Union[torch.Tensor, List[torch.Tensor]], *args, **kwargs) -> torch.Tensor:
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


class SwinTransformer(Backbone):

    def __init__(self, pretrained: bool = True, version: str = 'small_patch4_window7_224'):
        super().__init__()
        self._model: _SwinTransformer = create_model(model_name=f'swin_{version}', pretrained=pretrained)
        self._model.head = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model.forward_features(x)

    @property
    def num_features(self) -> int:
        return self._model.num_features

    @classmethod
    def get_label_prefix(cls) -> str:
        return 'swin_'


def create_image_backbone(label: str, pretrained: bool = True) -> Backbone:
    if label.startswith(ResNet.get_label_prefix()):
        return ResNet(pretrained=pretrained, version=label.replace(ResNet.get_label_prefix(), ''))
    elif label.startswith(EfficientNet.get_label_prefix()):
        return EfficientNet(pretrained=pretrained, version=label.replace(EfficientNet.get_label_prefix(), ''))
    elif label.startswith(SwinTransformer.get_label_prefix()):
        return SwinTransformer(pretrained=pretrained, version=label.replace(SwinTransformer.get_label_prefix(), ''))
    raise ValueError(f'Unsupported backbone: {label}.')


class LSTM(Backbone):

    def __init__(
            self,
            word_emb_dim: int,
            rnn_hidden_dim: int,
            num_features: int,
            rnn_num_layers: int = 1,
            dropout: float = 0.5):
        super().__init__()
        self._rnn = torch.nn.LSTM(
            input_size=word_emb_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            bidirectional=True)
        self._dropout = torch.nn.Dropout(dropout)
        self._fc = torch.nn.Linear(rnn_hidden_dim * 2, num_features)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x = torch.nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
        _, (out, _) = self._rnn(x)
        out = self._dropout(torch.cat((out[-2, :, :], out[-1, :, :]), dim=1))
        return self._fc(out)

    @property
    def num_features(self) -> int:
        return self._fc.out_features

    @classmethod
    def get_label_prefix(cls) -> str:
        return 'lstm'


def create_text_backbone(
        label: str,
        word_emb_dim: int,
        rnn_hidden_dim: int,
        num_features: int,
        rnn_num_layers: int = 1,
        dropout: float = 0.5) -> Backbone:
    if label.startswith(LSTM.get_label_prefix()):
        return LSTM(
            word_emb_dim=word_emb_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            num_features=num_features,
            rnn_num_layers=rnn_num_layers,
            dropout=dropout)
    raise ValueError(f'Unknown backbone label: {label}.')
