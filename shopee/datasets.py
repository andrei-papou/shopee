from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, List

import albumentations as alb
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset


class ImageAugmenter:

    def __init__(
            self,
            augmentation_list: Optional[List[alb.BasicTransform]] = None,
            img_size: Tuple[int, int] = (224, 224)):
        # noinspection PyTypeChecker
        self._augmentation = alb.Compose(
            [alb.Resize(*img_size, always_apply=True)] +
            (augmentation_list if augmentation_list is not None else []) +
            [alb.Normalize(), ToTensorV2()])

    def augment(self, image: np.ndarray) -> torch.Tensor:
        return self._augmentation(image=image)['image']


def load_image(img_path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)


class RandomTripletImageDataset(Dataset):

    def __init__(
            self,
            df: pd.DataFrame,
            image_folder_path: Path,
            augmentation_list: Optional[List[alb.BasicTransform]] = None):
        self._df = df
        self._image_folder_path = image_folder_path
        self._augmenter = ImageAugmenter(augmentation_list)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self._df.iloc[idx]
        a_img_name, a_lg_id = row['image'], row['label_group']
        p_img_name = self._df\
            .loc[(self._df.label_group == a_lg_id) & (self._df.image != a_img_name)]\
            .sample(n=1).image.item()
        n_img_name = self._df.loc[self._df.label_group != a_lg_id].sample(n=1).image.item()
        return (
            self._augmenter.augment(load_image(self._image_folder_path / a_img_name)),
            self._augmenter.augment(load_image(self._image_folder_path / p_img_name)),
            self._augmenter.augment(load_image(self._image_folder_path / n_img_name)))


class OnlineTripletImageDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            image_folder_path: Path,
            augmentation_list: Optional[List[alb.BasicTransform]] = None):
        self._df = df
        self._image_folder_path = image_folder_path
        self._augmenter = ImageAugmenter(augmentation_list)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self._df.iloc[idx]
        a_img_name, a_lg_id = row['image'], row['label_group']
        p_img_name = self._df \
            .loc[(self._df.label_group == a_lg_id) & (self._df.image != a_img_name)] \
            .sample(n=1).image.item()
        n_img_name = self._df.loc[self._df.label_group != a_lg_id].sample(n=1).image.item()
        return (
            self._augmenter.augment(load_image(self._image_folder_path / a_img_name)),
            self._augmenter.augment(load_image(self._image_folder_path / p_img_name)),
            self._augmenter.augment(load_image(self._image_folder_path / n_img_name)))


class ImageClsDataset(Dataset):

    def __init__(
            self,
            df: pd.DataFrame,
            image_folder_path: Path,
            augmentation_list: Optional[List[alb.BasicTransform]] = None,
            img_size: Tuple[int, int] = (224, 224)):
        self._df = df
        self._image_folder_path = image_folder_path
        self._augmenter = ImageAugmenter(augmentation_list, img_size=img_size)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self._df.iloc[idx]
        img_name, cls = row['image'], row['cls']  # type: (str, int)
        image = load_image(self._image_folder_path / img_name)
        return self._augmenter.augment(image), cls


class PrecomputedTripletImageDataset(Dataset):

    def __init__(
            self,
            df: pd.DataFrame,
            image_folder_path: Path,
            augmentation_list: Optional[List[alb.BasicTransform]] = None):
        self._df = df
        self._image_folder_path = image_folder_path
        self._augmenter = ImageAugmenter(augmentation_list)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self._df.iloc[idx]
        img_name_a, img_name_p, img_name_n = row['image_a'], row['image_p'], row['image_n']
        return (
            self._augmenter.augment(load_image(self._image_folder_path / img_name_a)),
            self._augmenter.augment(load_image(self._image_folder_path / img_name_p)),
            self._augmenter.augment(load_image(self._image_folder_path / img_name_n)))


class ImageTestPairDataset(Dataset):

    def __init__(
            self,
            df: pd.DataFrame,
            image_folder_path: Path,
            augmentation_list: Optional[List[alb.BasicTransform]] = None):
        self._df = df
        self._image_folder_path = image_folder_path
        self._augmenter = ImageAugmenter(augmentation_list)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self._df.iloc[idx]
        img_name_1, lg_1 = row['image_1'], row['label_group_1']
        img_name_2, lg_2 = row['image_2'], row['label_group_2']
        return \
            self._augmenter.augment(load_image(self._image_folder_path / img_name_1)), \
            self._augmenter.augment(load_image(self._image_folder_path / img_name_2)), \
            torch.tensor([int(lg_1 == lg_2)])


class PostingIdImageDataset(Dataset):

    def __init__(
            self,
            df: pd.DataFrame,
            image_folder_path: Path,
            augmentation_list: Optional[List[alb.BasicTransform]] = None):
        self._df = df
        self._image_folder_path = image_folder_path
        self._augmenter = ImageAugmenter(augmentation_list)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        row = self._df.iloc[idx]
        return row['posting_id'], self._augmenter.augment(load_image(self._image_folder_path / row['image']))


class ImageLabelGroupDataset(Dataset):

    def __init__(
            self,
            df: pd.DataFrame,
            image_folder_path: Path,
            augmentation_list: Optional[List[alb.BasicTransform]] = None):
        self._df = df
        self._image_folder_path = image_folder_path
        self._augmenter = ImageAugmenter(augmentation_list)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self._df.iloc[idx]
        return self._augmenter.augment(load_image(self._image_folder_path / row['image'])), row['label_group']
