from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import LightningLoggerBase
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from shopee.backbones import Backbone, EfficientNet
from shopee.checkpoint import create_checkpoint_callback
from shopee.datasets import RandomTripletImageDataset, PrecomputedTripletImageDataset
from shopee.module import Module, StepResult
from shopee.types import Triplet, TripletTuple, OptimizerConfig


def get_triplet_accuracy_components(t: Triplet, margin: float) -> Tuple[int, int]:
    av = torch.unsqueeze(t.a, dim=1)
    pv = torch.unsqueeze(t.p, dim=1)
    nv = torch.unsqueeze(t.n, dim=1)
    n_correct = (torch.cdist(av, nv) - torch.cdist(av, pv) >= margin).int().sum().item()
    return n_correct, av.shape[0]


class TripletModel(Module):

    def __init__(self, backbone: Backbone, lr: float = 1e-3, momentum: float = 0.9, margin: float = 1.0):
        super().__init__()
        self.backbone = backbone
        self.lr = lr
        self.momentum = momentum
        self.loss_fn = torch.nn.TripletMarginLoss(margin=margin)
        self.pooling = torch.nn.AdaptiveAvgPool2d(1)
        self.margin = margin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.backbone(x)
        return self.pooling(x).view(batch_size, -1)

    def _forward_triplet(self, triplet: Triplet) -> Triplet:
        return Triplet.from_first_dim_split(self(triplet.join()))

    def _step(self, batch: TripletTuple, batch_idx: int) -> StepResult:
        out = self._forward_triplet(Triplet.from_tuple(batch))
        loss = self.loss_fn(out.a, out.p, out.n)
        num_correct, num_total = get_triplet_accuracy_components(out, margin=self.margin)
        return {'loss': loss, 'num_correct': num_correct, 'num_total': num_total}

    def training_step(self, batch: TripletTuple, batch_idx: int) -> StepResult:
        return self._step(batch, batch_idx)

    def validation_step(self, batch: TripletTuple, batch_idx: int) -> StepResult:
        return self._step(batch, batch_idx)

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
        num_workers: int = 2,
        accumulate_grad_batches: int = 1,
        start_from_checkpoint_path: Optional[str] = None):
    data_root_path = Path(data_root_path)
    image_folder_path = data_root_path / 'train_images'

    index_root_path = Path(index_root_path)
    train_df = pd.read_csv(index_root_path / 'train-set.csv')
    test_pair_df = pd.read_csv(index_root_path / 'test_triplets.csv')

    train_dataset = RandomTripletImageDataset(df=train_df, image_folder_path=image_folder_path)
    valid_dataset = PrecomputedTripletImageDataset(df=test_pair_df, image_folder_path=image_folder_path)
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

    checkpoint_callback = create_checkpoint_callback(checkpoint_file_path=checkpoint_file_path)
    early_stopping_callback = EarlyStopping(
        monitor='valid_accuracy',
        mode='max',
        patience=max_epochs_no_improvement)
    backbone = EfficientNet(pretrained=start_from_checkpoint_path is None)
    model = TripletModel(backbone=backbone, lr=lr, momentum=momentum, margin=margin)
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


def load_model(checkpoint_path: str) -> TripletModel:
    return TripletModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        backbone=EfficientNet(pretrained=False)).cuda()
