import math
from pathlib import Path
from typing import Tuple, Optional

import albumentations as alb
import pandas as pd
import torch
import torch.nn.functional as torch_f
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import LightningLoggerBase
from timm import create_model
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from shopee.backbones import Backbone, EfficientNet
from shopee.checkpoint import create_checkpoint_callback
from shopee.datasets import ImageClsDataset
from shopee.module import Module, StepResult
from shopee.types import OptimizerConfig


def get_accuracy_components(logit: torch.Tensor, y: torch.Tensor) -> Tuple[int, int]:
    num_total: int = y.shape[0]
    y_pred: torch.Tensor = torch.argmax(torch_f.softmax(logit, dim=1), dim=1)
    num_correct: int = torch.eq(y, y_pred).int().sum().item()
    return num_correct, num_total


class ArcMarginProduct(torch.nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            scale: float = 30.0,
            margin: float = 0.50,
            easy_margin: float = False,
            ls_eps: float = 0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps  # label smoothing
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = torch_f.linear(torch_f.normalize(x), torch_f.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi: torch.Tensor = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(torch.gt(cosine, 0), phi, cosine)
        else:
            phi = torch.where(torch.gt(cosine, self.th), phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output


class ArcFaceModel(Module):

    def __init__(
            self,
            model_name: str,
            n_classes: int = 11014,
            fc_dim: int = 512,
            margin: float = 0.5,
            scale: float = 30.0,
            use_fc: bool = True,
            pretrained: bool = False,
            lr: float = 1e-3):
        super().__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = create_model(model_name, pretrained=pretrained)

        if model_name == 'resnext50_32x4d':
            final_in_features = self.backbone.fc.in_features
            self.apply_pooling = True
            self.backbone.fc = torch.nn.Identity()
            self.backbone.global_pool = torch.nn.Identity()

        elif model_name == 'efficientnet_b3':
            final_in_features = self.backbone.classifier.in_features
            self.apply_pooling = True
            self.backbone.classifier = torch.nn.Identity()
            self.backbone.global_pool = torch.nn.Identity()

        elif model_name == 'tf_efficientnet_b5_ns':
            final_in_features = self.backbone.classifier.in_features
            self.apply_pooling = True
            self.backbone.classifier = torch.nn.Identity()
            self.backbone.global_pool = torch.nn.Identity()

        elif model_name == 'eca_nfnet_l0':
            final_in_features = self.backbone.head.fc.in_features
            self.apply_pooling = True
            self.backbone.head.fc = torch.nn.Identity()
            self.backbone.head.global_pool = torch.nn.Identity()

        elif model_name == 'swin_small_patch4_window7_224':
            final_in_features = self.backbone.num_features
            self.apply_pooling = False
            self.backbone.head = torch.nn.Identity()

        else:
            raise ValueError(f'Unknown model_name = "{model_name}".')

        self.pooling = torch.nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc

        self.dropout = torch.nn.Dropout(p=0.0)
        self.fc = torch.nn.Linear(final_in_features, fc_dim)
        self.bn = torch.nn.BatchNorm1d(fc_dim)
        self._init_params()
        final_in_features = fc_dim

        self.final = ArcMarginProduct(
            final_in_features,
            n_classes,
            scale=scale,
            margin=margin,
            easy_margin=False,
            ls_eps=0.0
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.lr = lr

    def _init_params(self):
        torch.nn.init.xavier_normal_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.constant_(self.bn.weight, 1)
        torch.nn.init.constant_(self.bn.bias, 0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.extract_feat(image)
        return feature

    def extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.backbone(x)
        if self.apply_pooling:
            x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)
        return x

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> StepResult:
        x, y = batch
        logit = self.final(self(x), y)
        loss = self.loss_fn(logit, y)
        num_correct, num_total = get_accuracy_components(logit=logit, y=y)
        return {'loss': loss, 'num_correct': num_correct, 'num_total': num_total}

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> StepResult:
        return self._step(batch, batch_idx)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> StepResult:
        return self._step(batch, batch_idx)

    def configure_optimizers(self) -> OptimizerConfig:
        optimizer = Adam(self.parameters(), lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'train_loss',
        }


def train_model(
        index_root_path: str,
        data_root_path: str,
        checkpoint_file_path: str,
        lr: float = 1e-3,
        num_epochs: int = 20,
        margin: float = 0.5,
        max_epochs_no_improvement: int = 7,
        train_batch_size: int = 64,
        valid_batch_size: int = 64,
        num_workers: int = 2,
        img_size: Tuple[int, int] = (224, 224),
        logger: Optional[LightningLoggerBase] = None,
        train_index_file_name: Optional[str] = None,
        test_index_file_name: Optional[str] = None,
        start_from_checkpoint_path: Optional[str] = None,
        backbone_version: str = 'swin_small_patch4_window7_224',
        accumulate_grad_batches: int = 1,
        monitor_metric: str = 'train_loss',
        monitor_mode: str = 'min',
        check_val_every_n_epoch: int = 1000):
    data_root_path = Path(data_root_path)
    image_folder_path = data_root_path / 'train_images'

    index_root_path = Path(index_root_path)
    train_df = pd.read_csv(index_root_path / train_index_file_name)
    test_df = pd.read_csv(index_root_path / test_index_file_name)
    num_classes = len(train_df.cls.unique().tolist())

    train_dataset = ImageClsDataset(
        df=train_df,
        image_folder_path=image_folder_path,
        img_size=img_size,
        augmentation_list=[
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.Rotate(limit=120, p=0.8),
            alb.RandomBrightness(limit=(0.09, 0.6), p=0.5),
        ])
    valid_dataset = ImageClsDataset(
        df=test_df,
        image_folder_path=image_folder_path,
        img_size=img_size)
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True)
    valid_data_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=valid_batch_size,
        num_workers=num_workers,
        pin_memory=True)

    checkpoint_callback = create_checkpoint_callback(
        checkpoint_file_path=checkpoint_file_path,
        monitor=monitor_metric,
        mode=monitor_mode)
    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric,
        mode=monitor_mode,
        patience=max_epochs_no_improvement)
    model = ArcFaceModel(
        model_name=backbone_version,
        n_classes=num_classes,
        margin=margin,
        lr=lr)
    trainer = Trainer(
        auto_lr_find=True,
        gpus=1,
        auto_select_gpus=True,
        max_epochs=num_epochs,
        logger=logger,
        check_val_every_n_epoch=check_val_every_n_epoch,
        accumulate_grad_batches=accumulate_grad_batches,
        resume_from_checkpoint=start_from_checkpoint_path,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
        ])
    trainer.fit(
        model=model,
        train_dataloader=train_data_loader,
        val_dataloaders=valid_data_loader)


def load_model(
        checkpoint_path: str,
        n_classes: int = 11014,
        backbone_version: str = 'swin_small_patch4_window7_224') -> ArcFaceModel:
    return ArcFaceModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name=backbone_version,
        n_classes=n_classes).cuda()
