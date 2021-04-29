from pathlib import Path
from typing import Optional, Tuple

import albumentations as alb
import pandas as pd
import torch
import torch.nn.functional as torch_f
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import LightningLoggerBase
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from shopee.backbones import EfficientNet, Backbone
from shopee.checkpoint import create_checkpoint_callback
from shopee.datasets import ImageTestPairDataset, ImageLabelGroupDataset
from shopee.module import Module, StepResult
from shopee.pair_selection import get_label_similarity_matrix, get_hardest_triplet
from shopee.sampler import TripletOnlineSampler
from shopee.types import Triplet, OptimizerConfig


def get_accuracy_components(x1: torch.Tensor, x2: torch.Tensor, label: torch.Tensor) -> Tuple[int, int]:
    distance = torch_f.pairwise_distance(x1, x2)
    return


def get_triplet_accuracy_components(t: Triplet, margin: float) -> Tuple[int, int]:
    n: int = t.p.shape[0] + t.n.shape[0]
    av = torch.unsqueeze(t.a, dim=1)
    pv = torch.unsqueeze(t.p, dim=1)
    nv = torch.unsqueeze(t.n, dim=1)
    n_correct = (torch.cdist(av, pv) <= margin).int().sum().item() + \
        (torch.cdist(av, nv) > margin).int().sum().item()
    return n_correct, n


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin: float):
        super().__init__()
        self._margin = margin

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        euclidean_distance = torch_f.pairwise_distance(x1, x2)
        pos_error = torch.pow(euclidean_distance, 2)
        neg_error = torch.pow(torch.clamp(self._margin - euclidean_distance, min=0.0), 2)
        return torch.mean(label * pos_error + (1 - label) * neg_error)

    def forward_triplet(self, t: Triplet) -> torch.Tensor:
        return self(
            torch.cat([t.a, t.a], dim=0),
            torch.cat([t.p, t.n], dim=0),
            torch.cat([
                torch.ones((t.p.shape[0],)),
                torch.zeros((t.n.shape[0],))
            ]).cuda())


class ContrastiveModel(Module):

    def __init__(self, backbone: Backbone, lr: float = 1e-3, momentum: float = 0.9, margin: float = 1.0):
        super().__init__()
        self.model = backbone
        self.pooling = torch.nn.AdaptiveAvgPool2d(1)
        self.lr = lr
        self.momentum = momentum
        self.loss_fn = ContrastiveLoss(margin=margin)
        self.margin = margin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.model(x)
        return self.pooling(x).view(batch_size, -1)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> StepResult:
        xs, labels = batch
        embeds = self(xs)
        label_similarity_matrix = get_label_similarity_matrix(labels)
        out = get_hardest_triplet(embed_tensor=embeds, label_similarity_matrix=label_similarity_matrix)
        loss = self.loss_fn.forward_triplet(out)
        num_correct, num_total = get_triplet_accuracy_components(out, margin=self.margin)
        return {'loss': loss, 'num_correct': num_correct, 'num_total': num_total}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> StepResult:
        x1, x2, label = batch
        emb1, emb2 = self(x1), self(x2)
        loss = self.loss_fn(emb1, emb2, label)
        num_correct, num_total = get_triplet_accuracy_components(out, margin=self.margin)
        return {'loss': loss, 'num_correct': num_correct, 'num_total': num_total}

    def configure_optimizers(self) -> OptimizerConfig:
        optimizer = SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.1, patience=3, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'valid_accuracy',
        }


def train_model(
        index_root_path: str,
        data_root_path: str,
        checkpoint_file_path: str,
        logger: LightningLoggerBase,
        lr: float = 1e-3,
        momentum: float = 0.9,
        num_epochs: int = 25,
        margin: float = 1.0,
        max_epochs_no_improvement: int = 7,
        train_batch_size: int = 64,
        valid_batch_size: int = 64,
        accumulate_grad_batches: int = 1,
        train_index_file_name: Optional[str] = None,
        test_index_file_name: Optional[str] = None,
        num_workers: int = 2,
        num_train_batches: int = 6500,
        backbone_version: str = 'b1',
        start_from_checkpoint_path: Optional[str] = None):
    data_root_path = Path(data_root_path)
    image_folder_path = data_root_path / 'train_images'

    index_root_path = Path(index_root_path)
    train_df = pd.read_csv(index_root_path / (train_index_file_name or 'train-set.csv'))
    test_pair_df = pd.read_csv(index_root_path / (test_index_file_name or 'test_triplets.csv'))

    train_dataset = ImageLabelGroupDataset(
        df=train_df,
        image_folder_path=image_folder_path,
        augmentation_list=[
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.Rotate(limit=120, p=0.8),
            alb.RandomBrightness(limit=(0.09, 0.6), p=0.5),
        ])
    valid_dataset = ImageTestPairDataset(df=test_pair_df, image_folder_path=image_folder_path)
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=TripletOnlineSampler(df=train_df, batch_size=train_batch_size, num_batches=num_train_batches))
    valid_data_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=valid_batch_size,
        num_workers=num_workers,
        pin_memory=True)

    checkpoint_callback = create_checkpoint_callback(checkpoint_file_path=checkpoint_file_path)
    early_stopping_callback = EarlyStopping(
        monitor='valid_accuracy',
        mode='max',
        patience=max_epochs_no_improvement)
    backbone = EfficientNet(pretrained=start_from_checkpoint_path is None, version=backbone_version)
    model = ContrastiveModel(backbone=backbone, lr=lr, momentum=momentum, margin=margin)
    trainer = Trainer(
        auto_lr_find=True,
        gpus=1,
        auto_select_gpus=True,
        max_epochs=num_epochs,
        logger=logger,
        resume_from_checkpoint=start_from_checkpoint_path,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
        ])
    trainer.fit(
        model=model,
        train_dataloader=train_data_loader,
        val_dataloaders=valid_data_loader)


def load_model(checkpoint_path: str, backbone_version: str = 'b1') -> ContrastiveModel:
    return ContrastiveModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        backbone=EfficientNet(pretrained=False, version=backbone_version)).cuda()
