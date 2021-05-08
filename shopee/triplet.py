from pathlib import Path
from typing import Optional, Tuple, List, Union

import albumentations as alb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_f
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import LightningLoggerBase
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from shopee.backbones import Backbone, create_image_backbone, create_text_backbone
from shopee.checkpoint import create_checkpoint_callback
from shopee.datasets import PrecomputedTripletImageDataset, ImageLabelGroupDataset, TitleLabelGroupDataset, \
    PrecomputedTripletTitleDataset, collate_title_label, collate_title_triplet, read_translation_dict, \
    PIDTitleDataset, collate_pid_title
from shopee.module import Module, StepResult
from shopee.sampler import TripletOnlineSampler
from shopee.types import Triplet, TripletTuple, OptimizerConfig


class TripletDistance(torch.nn.Module):

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def get_distance_matrix(self, emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class EuclideanTripletDistance(TripletDistance):

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch_f.pairwise_distance(x1, x2)

    def get_distance_matrix(self, emb: torch.Tensor) -> torch.Tensor:
        n, _ = emb.shape
        norm_sq = torch.sum(emb ** 2, dim=1, keepdim=True).repeat(1, n)
        dot_prod = torch.matmul(emb, emb.T)
        return norm_sq - 2 * dot_prod + norm_sq.T


class CosineTripletDistance(TripletDistance):

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch_f.cosine_similarity(x1, x2)

    def get_distance_matrix(self, emb: torch.Tensor) -> torch.Tensor:
        emb = torch_f.normalize(emb, dim=1)
        return 1.0 - torch.matmul(emb, emb.T)


def get_triplet_distance(distance: str) -> TripletDistance:
    if distance == 'euclidean':
        return EuclideanTripletDistance()
    elif distance == 'cosine':
        return CosineTripletDistance()
    raise ValueError(f'Unknown distance: {distance}.')


def get_label_similarity_matrix(labels: torch.Tensor) -> torch.Tensor:
    rows = torch.unsqueeze(labels, dim=1).repeat(1, len(labels))
    cols = torch.unsqueeze(labels, dim=0).repeat(len(labels), 1)
    return torch.eq(rows, cols).int()


def get_hardest_triplet(
        embed_tensor: torch.Tensor,
        label_similarity_matrix: torch.Tensor,
        distance: TripletDistance) -> Triplet:
    dist_matrix = distance.get_distance_matrix(embed_tensor)

    hardest_pos = (dist_matrix * label_similarity_matrix).argmax(dim=1, keepdim=True)
    neg_masked = dist_matrix * label_similarity_matrix.logical_not().int()
    hardest_neg = (neg_masked + (neg_masked <= 0).int() * torch.finfo(torch.float32).max).argmin(dim=1, keepdim=True)

    return Triplet(
        a=embed_tensor,
        p=embed_tensor[hardest_pos, :].squeeze(dim=1),
        n=embed_tensor[hardest_neg, :].squeeze(dim=1))


def get_triplet_accuracy_components(t: Triplet, margin: float, distance: TripletDistance) -> Tuple[int, int]:
    n_correct = (distance(t.a, t.n) - distance(t.a, t.p) >= margin).int().sum().item()
    return n_correct, t.a.shape[0]


class TripletModel(Module):

    def __init__(
            self,
            backbone: Backbone,
            lr: float = 1e-3,
            momentum: float = 0.9,
            margin: float = 0.5,
            apply_pooling: bool = True,
            distance: str = 'euclidean'):
        super().__init__()
        self.backbone = backbone
        self.lr = lr
        self.momentum = momentum
        self.distance = get_triplet_distance(distance)
        self.loss_fn = torch.nn.TripletMarginWithDistanceLoss(
            margin=margin,
            distance_function=self.distance)
        self.pooling = torch.nn.AdaptiveAvgPool2d(1) if apply_pooling else None
        self.margin = margin

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        batch_size = x.shape[0] if isinstance(x, torch.Tensor) else len(x)
        x = self.backbone(x)
        if self.pooling is not None:
            x = self.pooling(x)
            x = x.view(batch_size, -1)
        return torch_f.normalize(x, dim=1)

    def _forward_triplet(self, triplet: Triplet) -> Triplet:
        return Triplet(a=self(triplet.a), p=self(triplet.p), n=self(triplet.n))

    def training_step(
            self,
            batch: Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor],
            batch_idx: int) -> StepResult:
        xs, labels = batch
        embeds = self(xs)
        label_similarity_matrix = get_label_similarity_matrix(labels)
        out = get_hardest_triplet(
            embed_tensor=embeds, label_similarity_matrix=label_similarity_matrix, distance=self.distance)
        loss = self.loss_fn(out.a, out.p, out.n)
        num_correct, num_total = get_triplet_accuracy_components(out, margin=self.margin, distance=self.distance)
        return {'loss': loss, 'num_correct': num_correct, 'num_total': num_total}

    def validation_step(self, batch: TripletTuple, batch_idx: int) -> StepResult:
        out = self._forward_triplet(Triplet.from_tuple(batch))
        loss = self.loss_fn(out.a, out.p, out.n)
        num_correct, num_total = get_triplet_accuracy_components(out, margin=self.margin, distance=self.distance)
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
        margin: float = 0.5,
        max_epochs_no_improvement: int = 7,
        train_batch_size: int = 64,
        valid_batch_size: int = 64,
        num_workers: int = 2,
        accumulate_grad_batches: int = 1,
        num_train_batches: int = 6500,
        start_from_checkpoint_path: Optional[str] = None,
        train_index_file_name: Optional[str] = None,
        test_index_file_name: Optional[str] = None,
        img_size: Tuple[int, int] = (224, 224),
        backbone_label: str = 'efficientnetb1',
        distance: str = 'euclidean',
        gpus: Optional[int] = None,
        tpus: Optional[int] = None):
    data_root_path = Path(data_root_path)
    image_folder_path = data_root_path / 'train_images'

    index_root_path = Path(index_root_path)
    train_df = pd.read_csv(index_root_path / (train_index_file_name or 'train-set.csv'))
    test_pair_df = pd.read_csv(index_root_path / (test_index_file_name or 'test_triplets.csv'))

    train_dataset = ImageLabelGroupDataset(
        df=train_df,
        image_folder_path=image_folder_path,
        img_size=img_size,
        augmentation_list=[
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.Rotate(limit=120, p=0.8),
            alb.RandomBrightness(limit=(0.09, 0.6), p=0.5),
        ])
    valid_dataset = PrecomputedTripletImageDataset(
        df=test_pair_df,
        image_folder_path=image_folder_path,
        img_size=img_size)
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
    backbone = create_image_backbone(label=backbone_label, pretrained=start_from_checkpoint_path is None)
    model = TripletModel(backbone=backbone, lr=lr, momentum=momentum, margin=margin, distance=distance)
    trainer = Trainer(
        auto_lr_find=True,
        gpus=gpus,
        auto_select_gpus=gpus is not None,
        tpu_cores=tpus,
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


def load_model(checkpoint_path: str, backbone_label: str = 'efficientnetb1') -> TripletModel:
    return TripletModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        backbone=create_image_backbone(pretrained=False, label=backbone_label)).cuda()


def train_text_model(
        index_root_path: str,
        checkpoint_file_path: str,
        logger: LightningLoggerBase,
        lr: float = 1e-3,
        momentum: float = 0.9,
        num_epochs: int = 25,
        margin: float = 0.5,
        max_epochs_no_improvement: int = 7,
        train_batch_size: int = 64,
        valid_batch_size: int = 64,
        num_workers: int = 2,
        accumulate_grad_batches: int = 1,
        num_train_batches: int = 6500,
        start_from_checkpoint_path: Optional[str] = None,
        train_index_file_name: Optional[str] = None,
        test_index_file_name: Optional[str] = None,
        backbone_label: str = 'lstm',
        distance: str = 'cosine',
        gpus: Optional[int] = None,
        tpus: Optional[int] = None,
        translation_dict_path: str = 'resources/indonesian_to_english.json'):
    index_root_path = Path(index_root_path)
    train_df = pd.read_csv(index_root_path / (train_index_file_name or 'train-set.csv'))
    test_pair_df = pd.read_csv(index_root_path / (test_index_file_name or 'test_triplets.csv'))
    translation_dict = read_translation_dict(translation_dict_path) if translation_dict_path is not None else None

    train_dataset = TitleLabelGroupDataset(df=train_df, translation_dict=translation_dict)
    valid_dataset = PrecomputedTripletTitleDataset(df=test_pair_df, translation_dict=translation_dict)
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=TripletOnlineSampler(df=train_df, batch_size=train_batch_size, num_batches=num_train_batches),
        collate_fn=collate_title_label)
    valid_data_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=valid_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_title_triplet)

    checkpoint_callback = create_checkpoint_callback(checkpoint_file_path=checkpoint_file_path)
    early_stopping_callback = EarlyStopping(
        monitor='valid_accuracy',
        mode='max',
        patience=max_epochs_no_improvement)
    backbone = create_text_backbone(
        label=backbone_label,
        word_emb_dim=300,
        rnn_hidden_dim=256,
        num_features=256)
    model = TripletModel(
        backbone=backbone,
        lr=lr,
        momentum=momentum,
        margin=margin,
        distance=distance,
        apply_pooling=False)
    trainer = Trainer(
        auto_lr_find=True,
        gpus=gpus,
        auto_select_gpus=gpus is not None,
        tpu_cores=tpus,
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


def load_text_model(
        checkpoint_path: str,
        backbone_label: str,
        word_emb_dim: int,
        rnn_hidden_dim: int,
        num_features: int,
        rnn_num_layers: int = 1,
        dropout: float = 0.2) -> TripletModel:
    return TripletModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        backbone=create_text_backbone(
            label=backbone_label,
            word_emb_dim=word_emb_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            num_features=num_features,
            rnn_num_layers=rnn_num_layers,
            dropout=dropout),
        distance='cosine',
        apply_pooling=False).cuda()


def get_text_embedding_tuple(
        index_root_path: str,
        checkpoint_file_path: str,
        batch_size: int = 64,
        num_workers: int = 8,
        backbone_label: str = 'lstm',
        preprocess: bool = True,
        translation_dict_path: str = 'resources/indonesian_to_english.json',
        progress_bar: bool = False) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(index_root_path)
    model = load_text_model(
        checkpoint_path=checkpoint_file_path,
        backbone_label=backbone_label,
        word_emb_dim=300,
        rnn_hidden_dim=256,
        num_features=256,
        rnn_num_layers=1,
        dropout=0.2).cuda()
    model.eval()
    translation_dict = read_translation_dict(translation_dict_path) if translation_dict_path is not None else None
    dataset = PIDTitleDataset(df=df, preprocess=preprocess, translation_dict=translation_dict)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_pid_title,
        num_workers=num_workers,
        pin_memory=True)

    embedding_list: List[torch.Tensor] = []
    posting_id_list: List[str] = []
    with torch.no_grad():
        it = tqdm(data_loader, desc='embedding dict generation') if progress_bar else data_loader
        for pid_list, x_list in it:
            embedding_list.append(model([x.cuda() for x in x_list]).cpu())
            posting_id_list.extend(pid_list)
    return torch.cat(embedding_list, dim=0).numpy(), posting_id_list
