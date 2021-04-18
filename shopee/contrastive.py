from pathlib import Path
from typing import Optional, List

import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import LightningLoggerBase
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from shopee.backbones import ResNet18, Backbone
from shopee.checkpoint import create_checkpoint_callback
from shopee.datasets import RandomTripletImageDataset, PrecomputedTripletImageDataset
from shopee.evaluation import evaluate_generic_model
from shopee.metrics import get_triplet_accuracy_components
from shopee.module import Module, StepResult
from shopee.types import Triplet, TripletTuple, OptimizerConfig


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
        ones = torch.ones(*dist.shape).cuda()
        zeros = torch.zeros(*dist.shape).cuda()
        dist_to_margin = torch.max(ones * self._margin - dist, zeros)

        loss = dist * same_mask + dist_to_margin * diff_mask
        return loss.mean()

    def forward_triplet(self, t: Triplet) -> torch.Tensor:
        n = t.a.shape[0]
        pos_loss = self(t.a, t.p, torch.ones(n).cuda())
        neg_loss = self(t.a, t.n, torch.zeros(n).cuda())
        return (pos_loss + neg_loss) / 2


class ContrastiveModel(Module):

    def __init__(self, backbone: Backbone, lr: float = 1e-3, momentum: float = 0.9, margin: float = 1.0):
        super().__init__()
        self.model = backbone
        self.lr = lr
        self.momentum = momentum
        self.loss_fn = ContrastiveLoss(margin=margin)
        self.margin = margin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _forward_triplet(self, triplet: Triplet) -> Triplet:
        return Triplet.from_first_dim_split(self(triplet.join()))

    def _step(self, batch: TripletTuple, batch_idx: int) -> StepResult:
        out = self._forward_triplet(Triplet.from_tuple(batch))
        loss = self.loss_fn.forward_triplet(out)
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
        accumulate_grad_batches: int = 1,
        num_workers: int = 2,
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
    backbone = ResNet18(pretrained=start_from_checkpoint_path is None)
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


def evaluate_model(
        index_root_path: str,
        data_root_path: str,
        checkpoint_path: str,
        margin_list: List[float],
        batch_size: int = 64):
    model = ContrastiveModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        backbone=ResNet18(pretrained=False)).cuda()
    return evaluate_generic_model(
        model=model,
        index_root_path=index_root_path,
        data_root_path=data_root_path,
        margin_list=margin_list,
        batch_size=batch_size)
