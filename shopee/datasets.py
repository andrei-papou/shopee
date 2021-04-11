from pathlib import Path
from typing import Tuple, Dict, Sequence

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


_TRANSFORM_RESIZE = transforms.Resize(size=(224, 224))
_TRANSFORM_NORMALIZE = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def load_image(img_path: Path) -> torch.Tensor:
    img = torch.from_numpy(cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
    return _TRANSFORM_NORMALIZE(_TRANSFORM_RESIZE(img) / 255.0)


class RandomTripletImageDataset(Dataset):

    def __init__(self, df: pd.DataFrame, image_folder_path: Path):
        self._df = df
        self._image_folder_path = image_folder_path

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
            load_image(self._image_folder_path / a_img_name),
            load_image(self._image_folder_path / p_img_name),
            load_image(self._image_folder_path / n_img_name))


class ImageLabelDataset(Dataset):

    def __init__(self, df: pd.DataFrame, image_folder_path: Path):
        self._df = df
        self._image_folder_path = image_folder_path

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self._df.iloc[idx]
        img_name, label_group = row['image'], row['label_group']  # type: (str, int)
        return load_image(self._image_folder_path / img_name), label_group


class PrecomputedTripletImageDataset(Dataset):

    def __init__(self, df: pd.DataFrame, image_folder_path: Path):
        self._df = df
        self._image_folder_path = image_folder_path

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self._df.iloc[idx]
        img_name_a, img_name_p, img_name_n = row['image_a'], row['image_p'], row['image_n']
        return (
            load_image(self._image_folder_path / img_name_a),
            load_image(self._image_folder_path / img_name_p),
            load_image(self._image_folder_path / img_name_n))


class ImageTestPairDataset(Dataset):

    def __init__(self, df: pd.DataFrame, image_folder_path: Path):
        self._df = df
        self._image_folder_path = image_folder_path

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self._df.iloc[idx]
        img_name_1, lg_1 = row['image_1'], row['label_group_1']
        img_name_2, lg_2 = row['image_2'], row['label_group_2']
        return load_image(self._image_folder_path / img_name_1), \
            load_image(self._image_folder_path / img_name_2), \
            torch.tensor([int(lg_1 == lg_2)])


class PostingIdImageDataset(Dataset):

    def __init__(self, df: pd.DataFrame, image_folder_path: Path):
        self._df = df
        self._image_folder_path = image_folder_path

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        row = self._df.iloc[idx]
        return row['posting_id'], load_image(self._image_folder_path / row['image'])


class EmbeddingDataset(Dataset):

    def __init__(self, embedding_dict: Dict[str, torch.Tensor]):
        self._posting_id_list = [pi for pi in embedding_dict.keys()]
        self._embedding_dict = embedding_dict

    def __len__(self) -> int:
        return len(self._posting_id_list)

    def __getitem__(self, idx: int) -> Tuple[Sequence[str], torch.Tensor]:
        posting_id = self._posting_id_list[idx]
        return posting_id, self._embedding_dict[posting_id]
