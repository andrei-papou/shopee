from pathlib import Path
from statistics import mean
from typing import Dict, Set, Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from shopee.datasets import PostingIdImageDataset, EmbeddingDataset


def get_embedding_dict(
        df: pd.DataFrame,
        model: torch.nn.Module,
        data_path: Path,
        batch_size: int = 64,
        num_workers: int = 4,
        progress_bar: bool = False) -> Dict[str, torch.Tensor]:
    dataset = PostingIdImageDataset(df, data_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)

    embedding_dict: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        it = tqdm(data_loader, desc='embedding dict generation') if progress_bar else data_loader
        for posting_id_list, x in it:
            x = x.cuda()
            embedding_tensor = model(x)
            for posting_id, embedding in zip(posting_id_list, embedding_tensor):
                embedding_dict[posting_id] = embedding.cpu()
        del model
    return embedding_dict


def get_true_matches_dict(df: pd.DataFrame, progress_bar: bool = False) -> Dict[str, Set[str]]:
    matches_dict = {}
    row_iter = df.iterrows()
    it = tqdm(row_iter, total=len(df), desc='true matches dict generation') if progress_bar else row_iter
    for _, row in it:
        pid, lg = row['posting_id'], row['label_group']
        matches_dict[pid] = {p for p in df[df.label_group == lg].posting_id.unique().tolist()}
    return matches_dict


def get_pred_matches_dict(
        embedding_dict: Dict[str, torch.Tensor],
        threshold: float = 1.0,
        progress_bar: bool = False,
        batch_size: int = 1000,
        num_workers: int = 4) -> Dict[str, Set[str]]:
    matches_dict: Dict[str, Set[str]] = {}
    outer_embedding_dataset = EmbeddingDataset(embedding_dict)
    inner_embedding_dataset = EmbeddingDataset(embedding_dict)
    outer_data_loader = DataLoader(
        outer_embedding_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers)
    inner_data_loader = DataLoader(
        inner_embedding_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers)
    with torch.no_grad():
        it = tqdm(outer_data_loader, desc='pred matches dict generation') if progress_bar else outer_data_loader
        for outer_pid_list, outer_emb_tensor in it:
            outer_pid_match_dict: Dict[str, List[Tuple[str, float]]] = {pid: [] for pid in outer_pid_list}
            outer_emb_tensor = outer_emb_tensor.unsqueeze(0).cuda()
            for inner_pid_list, inner_emb_tensor in inner_data_loader:
                inner_emb_tensor = inner_emb_tensor.unsqueeze(0).cuda()
                dist_matrix = torch.cdist(outer_emb_tensor, inner_emb_tensor).squeeze(0)
                same_indice_seq = (dist_matrix <= threshold).int().nonzero(as_tuple=False)
                for same_idx_pair in same_indice_seq:
                    oi, ii = same_idx_pair[0].item(), same_idx_pair[1].item()
                    outer_pid_match_dict[outer_pid_list[oi]].append((inner_pid_list[ii], dist_matrix[oi, ii].item()))
            matches_dict.update({
                posting_id: {pid for (pid, _) in sorted(posting_id_match_list, key=lambda x: x[1])[:50]}
                for posting_id, posting_id_match_list in outer_pid_match_dict.items()
            })
    return matches_dict


def get_f1_mean_for_matches(
        true_matches_dict: Dict[str, Set[str]],
        pred_matches_dict: Dict[str, Set[str]]) -> float:
    pid_list = list(true_matches_dict.keys())
    p_bar = tqdm(pid_list)
    f1_val_list, true_match_count_list, pred_match_count_list = [], [], []
    for pid in p_bar:
        pred_match_pid_set = pred_matches_dict[pid]
        true_match_pid_set = true_matches_dict[pid]
        true_match_mask = np.array([int(pid in true_match_pid_set) for pid in pid_list])
        pred_match_mask = np.array([int(pid in pred_match_pid_set) for pid in pid_list])
        f1_val_list.append(f1_score(true_match_mask, pred_match_mask))
        true_match_count_list.append(len(true_match_pid_set))
        pred_match_count_list.append(len(pred_match_pid_set))
        p_bar.set_description(
            'f1_mean calculation, '
            f'score: {mean(f1_val_list):.5f}, '
            f'n_true: {mean(true_match_count_list):.2f}, '
            f'n_pred: {mean(pred_match_count_list):.2f}')
    return mean(f1_val_list)
