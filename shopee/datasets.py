from __future__ import annotations

import json
import re
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Tuple, Optional, List, Dict, TypedDict, NamedTuple, Counter as CounterT

import albumentations as alb
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from torchtext.vocab import Vocab, Vectors, FastText
from transformers import PreTrainedTokenizer, AutoTokenizer


class DataType(str, Enum):
    image = 'image'
    text = 'text'


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
            img_size: Tuple[int, int] = (224, 224),
            augmentation_list: Optional[List[alb.BasicTransform]] = None):
        self._df = df
        self._image_folder_path = image_folder_path
        self._augmenter = ImageAugmenter(augmentation_list, img_size=img_size)

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
            img_size: Tuple[int, int] = (224, 224),
            augmentation_list: Optional[List[alb.BasicTransform]] = None):
        self._df = df
        self._image_folder_path = image_folder_path
        self._augmenter = ImageAugmenter(augmentation_list, img_size=img_size)

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
            img_size: Tuple[int, int] = (224, 224),
            augmentation_list: Optional[List[alb.BasicTransform]] = None):
        self._df = df
        self._image_folder_path = image_folder_path
        self._augmenter = ImageAugmenter(augmentation_list, img_size=img_size)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self._df.iloc[idx]
        return self._augmenter.augment(load_image(self._image_folder_path / row['image'])), row['label_group']


DIGIT_PATTERN = '(\d+(\.\d*)?)'
ALPHA_PATTERN = '[a-zA-Z]+'
UNITS_PATTERN = '[a-wyzA-WYZ]+'
UNICODE_RE = r'((\\x[a-zA-Z0-9]{2}){2,4})'
digit_alpha_re = re.compile(f'^(?P<digit>{DIGIT_PATTERN})(?P<alpha>{ALPHA_PATTERN})$')
alpha_digit_re = re.compile(f'^(?P<alpha>{ALPHA_PATTERN})(?P<digit>{DIGIT_PATTERN})$')
digit_alpha_digit_re = re.compile(
    f'^(?P<digit1>{DIGIT_PATTERN})(?P<alpha>{ALPHA_PATTERN})(?P<digit2>{DIGIT_PATTERN})$')
alpha_digit_alpha_re = re.compile(
    f'^(?P<alpha1>{ALPHA_PATTERN})(?P<digit>{DIGIT_PATTERN})(?P<alpha2>{ALPHA_PATTERN})$')
size_2d_re = re.compile(
    f'^(?P<width_val>{DIGIT_PATTERN})(?P<width_unit>{UNITS_PATTERN})x'
    f'(?P<length_val>{DIGIT_PATTERN})(?P<length_unit>{UNITS_PATTERN})$')
size_3d_re = re.compile(
    f'^(?P<width_val>{DIGIT_PATTERN})(?P<width_unit>{UNITS_PATTERN})x'
    f'(?P<length_val>{DIGIT_PATTERN})(?P<length_unit>{UNITS_PATTERN})x'
    f'(?P<height_val>{DIGIT_PATTERN})(?P<height_unit>{UNITS_PATTERN})$')


def preprocess_alpha_token(tok: str) -> str:
    tok = tok.strip()
    if tok == 'g':
        return 'gr'
    return tok


def read_translation_dict(path: str) -> Dict[str, str]:
    result = {}
    with open(path) as f:
        word_list = json.load(f)
        for word in word_list:
            for ind_tok, eng_tok in word.items():
                result[ind_tok.lower()] = eng_tok.lower()
    return result


def preprocess_title(title: str, ind_to_eng_dict: Optional[Dict[str, str]] = None) -> str:
    title = title.lower() \
        .replace('=', '') \
        .replace(',', '') \
        .replace('&', '') \
        .replace('!', '') \
        .replace('#', '') \
        .replace('(', ' ') \
        .replace(')', ' ') \
        .replace('[', ' ') \
        .replace(']', ' ') \
        .replace('/', ' ') \
        .replace('|', ' ') \
        .replace('_', ' ') \
        .replace('{', ' ') \
        .replace('}', ' ') \
        .replace('-', ' ') \
        .replace('\\xc3\\x97', 'x')
    for x, _ in re.findall(UNICODE_RE, title):
        title = title.replace(x, '')
    if title.startswith('b"'):
        title = title[2:]
    if title.endswith('"'):
        title = title[:-1]
    title_part_list = title.split(' ')
    new_title_part_list = []
    for title_part in title_part_list:
        if not title_part.replace(' ', ''):
            continue

        if title_part.startswith('+'):
            title_part = f'+ {title_part[1:]}'
        if title_part.endswith('+'):
            title_part = f'{title_part[:-1]} +'
        m = digit_alpha_re.match(title_part)
        if m is not None:
            title_part = m.group('digit').strip() + ' ' + preprocess_alpha_token(m.group('alpha').strip())
        m = alpha_digit_re.match(title_part)
        if m is not None:
            title_part = preprocess_alpha_token(m.group('alpha').strip()) + ' ' + m.group('digit').strip()
        m = digit_alpha_digit_re.match(title_part)
        if m is not None:
            title_part = m.group('digit1').strip() + ' ' + \
                         preprocess_alpha_token(m.group('alpha').strip()) + ' ' + m.group('digit2').strip()
        m = alpha_digit_alpha_re.match(title_part)
        if m is not None:
            title_part = preprocess_alpha_token(m.group('alpha1').strip()) + ' ' + \
                         m.group('digit').strip() + ' ' + preprocess_alpha_token(m.group('alpha2').strip())
        m = size_2d_re.match(title_part)
        if m is not None:
            title_part = \
                m.group('width_val').strip() + ' ' + preprocess_alpha_token(m.group('width_unit').strip()) + \
                ' x ' + \
                m.group('length_val').strip() + ' ' + preprocess_alpha_token(m.group('length_unit').strip())
        m = size_3d_re.match(title_part)
        if m is not None:
            title_part = \
                m.group('width_val').strip() + ' ' + preprocess_alpha_token(m.group('width_unit').strip()) + \
                ' x ' + \
                m.group('length_val').strip() + ' ' + preprocess_alpha_token(m.group('length_unit').strip()) + \
                ' x ' + \
                m.group('height_val').strip() + ' ' + preprocess_alpha_token(m.group('height_unit').strip())
        title_part = title_part.strip()
        if title_part.endswith('.'):
            title_part = title_part[:-1]
        if ind_to_eng_dict is not None and title_part in ind_to_eng_dict:
            title_part = ind_to_eng_dict[title_part]
        new_title_part_list.append(title_part)
    return ' '.join(new_title_part_list)


def collate_pid_title(batch: List[Tuple[str, torch.Tensor]]) -> Tuple[List[str], List[torch.Tensor]]:
    pid_list, emb_list = [], []
    for (pid, emb) in batch:
        pid_list.append(pid)
        emb_list.append(emb)
    return pid_list, emb_list


class PIDTitleRecord(NamedTuple):
    pid: str
    title: str


class PIDTitleDataset(Dataset):

    def __init__(
            self,
            df: pd.DataFrame,
            preprocess: bool = True,
            translation_dict: Optional[Dict[str, str]] = None):
        index: List[PIDTitleRecord] = []
        for _, row in df.iterrows():
            title = preprocess_title(row['title'], translation_dict) if preprocess else row['title']
            index.append(PIDTitleRecord(pid=row['posting_id'], title=title))
        self._index = index
        self._vocab = FastText()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        row = self._index[idx]
        title_tensor = torch.stack([self._vocab[token] for token in row.title.split(' ')], dim=0)
        return row.pid, title_tensor


class PostingIdTitleDataset(Dataset):

    def __init__(
            self,
            df: pd.DataFrame,
            tokenizer_path: str,
            preprocess: bool = True,
            translation_dict: Optional[Dict[str, str]] = None):
        df['title'] = df.apply(
            lambda row: preprocess_title(row['title'], translation_dict if preprocess else row['title']), axis=1)
        self._df = df
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        row = self._df.iloc[idx]
        pid, title = row['posting_id'], row['title']
        title_enc = self._tokenizer(title, padding='max_length', truncation=True, max_length=16, return_tensors="pt")
        input_ids = title_enc['input_ids'][0]
        attention_mask = title_enc['attention_mask'][0]
        return pid, input_ids, attention_mask


def collate_title_label(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    x_list, y_list = [], []
    for (x, y) in batch:
        x_list.append(x)
        y_list.append(y)
    return x_list, torch.tensor(y_list)


class TitleLabelRecord(NamedTuple):
    title: str
    label_group: int


class TitleLabelGroupDataset(Dataset):

    def __init__(
            self,
            df: pd.DataFrame,
            preprocess: bool = True,
            translation_dict: Optional[Dict[str, str]] = None):
        index: List[TitleLabelRecord] = []
        for _, row in df.iterrows():
            title = preprocess_title(row['title'], translation_dict) if preprocess else row['title']
            index.append(TitleLabelRecord(title=title, label_group=row['label_group']))
        self._index = index
        self._vocab = FastText()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self._index[idx]
        title_tensor = torch.stack([self._vocab[token] for token in row.title.split(' ')], dim=0)
        return title_tensor, row.label_group


def collate_title_triplet(
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    a_list, p_list, n_list = [], [], []
    for (a, p, n) in batch:
        a_list.append(a)
        p_list.append(p)
        n_list.append(n)
    return a_list, p_list, n_list


class TitleTripletRecord(NamedTuple):
    title_a: str
    title_p: str
    title_n: str


class PrecomputedTripletTitleDataset(Dataset):

    def __init__(
            self,
            df: pd.DataFrame,
            preprocess: bool = True,
            translation_dict: Optional[Dict[str, str]] = None):
        index: List[TitleTripletRecord] = []

        for _, row in df.iterrows():
            title_a = preprocess_title(row['title_a'], translation_dict) if preprocess else row['title_a']
            title_p = preprocess_title(row['title_p'], translation_dict) if preprocess else row['title_p']
            title_n = preprocess_title(row['title_n'], translation_dict) if preprocess else row['title_n']
            index.append(TitleTripletRecord(title_a=title_a, title_p=title_p, title_n=title_n))

        self._index = index
        self._vocab = FastText()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self._index[idx]
        title_a_tensor = torch.stack([self._vocab[token] for token in row.title_a.split(' ')], dim=0)
        title_p_tensor = torch.stack([self._vocab[token] for token in row.title_p.split(' ')], dim=0)
        title_n_tensor = torch.stack([self._vocab[token] for token in row.title_n.split(' ')], dim=0)
        return title_a_tensor, title_p_tensor, title_n_tensor
