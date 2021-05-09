import pickle
from pathlib import Path
from statistics import mean
from typing import Dict, Set, Tuple, List, Union, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm import tqdm

from shopee.datasets import PostingIdImageDataset
from shopee.threshold import ThresholdFinder
from shopee.postprocessing import build_transitive_relations, postprocess_matches_dict

DEFAULT_DISTANCE_METRIC = 'minkowski'


def get_embedding_tuple(
        df: pd.DataFrame,
        model: torch.nn.Module,
        data_path: Path,
        batch_size: int = 64,
        num_workers: int = 4,
        img_size: Tuple[int, int] = (224, 224),
        progress_bar: bool = False) -> Tuple[np.ndarray, List[str]]:
    dataset = PostingIdImageDataset(df, data_path, img_size=img_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)

    embedding_list: List[torch.Tensor] = []
    posting_id_list: List[str] = []
    with torch.no_grad():
        it = tqdm(data_loader, desc='embedding dict generation') if progress_bar else data_loader
        for pid_list, x in it:
            y = model(x.cuda()).cpu()
            # print(y.shape)
            embedding_list.append(y)
            posting_id_list.extend(pid_list)
    return torch.cat(embedding_list, dim=0).numpy(), posting_id_list


def save_embedding_tuple(embedding_matrix: np.ndarray, posting_id_list: List[str], path: str):
    with open(path, 'wb') as f:
        pickle.dump((embedding_matrix, posting_id_list), f)


def load_embedding_tuple(path: str) -> Tuple[np.ndarray, List[str]]:
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_true_matches_dict(df: pd.DataFrame, progress_bar: bool = False) -> Dict[str, Set[str]]:
    matches_dict = {}
    row_iter = df.iterrows()
    it = tqdm(row_iter, total=len(df), desc='true matches dict generation') if progress_bar else row_iter
    for _, row in it:
        pid, lg = row['posting_id'], row['label_group']
        matches_dict[pid] = {p for p in df[df.label_group == lg].posting_id.unique().tolist()}
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


def get_distance_tuple(
        embedding_matrix: np.ndarray,
        max_matches: int = 50,
        distance_metric: str = DEFAULT_DISTANCE_METRIC) -> Tuple[np.ndarray, np.ndarray]:
    knn = NearestNeighbors(n_neighbors=max_matches, metric=distance_metric)
    knn.fit(embedding_matrix)
    distances, indices = knn.kneighbors(embedding_matrix)

    return distances, indices


def get_distance_pred_matches_dict(
        distance_matrix: np.ndarray,
        index_matrix: np.ndarray,
        posting_id_list: List[str],
        threshold: float,
        progress_bar: bool = False) -> Dict[str, Set[str]]:
    matches_dict: Dict[str, Set[str]] = {}
    n_total = distance_matrix.shape[0]
    it = range(n_total)
    for i in (tqdm(it, total=n_total, desc='distance pred matches dict generation') if progress_bar else it):
        local_match_index_arr: np.ndarray = np.where(distance_matrix[i] < threshold)[0]
        global_match_index_list = index_matrix[i, local_match_index_arr].tolist()
        matches_dict[posting_id_list[i]] = {posting_id_list[i] for i in global_match_index_list}
    return matches_dict


def get_phash_pred_matches_dict(df: pd.DataFrame, progress_bar: bool = False) -> Dict[str, Set[str]]:
    it = tqdm(df.iterrows(), total=len(df), desc='phash pred matches dict generation') \
        if progress_bar else df.iterrows()
    matches_dict = {}
    for _, row in it:
        pid, phash = row['posting_id'], row['image_phash']
        matches = {pid for pid in df[df.image_phash == phash].posting_id.tolist()}
        if isinstance(it, tqdm):
            it.set_description(f'phash pred matches dict generation, n={len(matches)}')
        matches_dict[pid] = matches
    return matches_dict


def join_matches_dict_list(matches_dict_list: List[Dict[str, Set[str]]]) -> Dict[str, Set[str]]:
    assert len(matches_dict_list) > 0
    pid_list = list(matches_dict_list[0].keys())
    join_matches_dict = {}
    for pid in pid_list:
        joined_matches = set()
        for matches_dict in matches_dict_list:
            joined_matches |= matches_dict[pid]
        join_matches_dict[pid] = joined_matches
    return join_matches_dict


def evaluate_model(
        model: torch.nn.Module,
        index_root_path: str,
        data_root_path: str,
        threshold: Union[ThresholdFinder, float],
        batch_size: int = 64,
        use_phash: bool = True,
        test_set_file_name: str = 'test-set.csv',
        distance_metric: str = DEFAULT_DISTANCE_METRIC):
    model.eval()
    eval_df = pd.read_csv(Path(index_root_path) / test_set_file_name)
    image_folder_path = Path(data_root_path) / 'train_images'

    with torch.no_grad():
        embedding_matrix, posting_id_list = get_embedding_tuple(
            df=eval_df,
            model=model,
            data_path=image_folder_path,
            batch_size=batch_size,
            progress_bar=True)
    evaluate_embeddings(
        embedding_tuple=(embedding_matrix, posting_id_list),
        index_root_path=index_root_path,
        threshold=threshold,
        use_phash=use_phash,
        distance_metric=distance_metric,
    )


def evaluate_embeddings(
        embedding_tuple: Tuple[np.ndarray, List[str]],
        index_root_path: str,
        threshold: Union[ThresholdFinder, float],
        use_phash: bool = True,
        test_set_file_name: str = 'test-set.csv',
        distance_metric: str = DEFAULT_DISTANCE_METRIC,
        transitive_relations_rounds: int = 0,
        postprocess: bool = False):
    eval_df = pd.read_csv(Path(index_root_path) / test_set_file_name)

    embedding_matrix, posting_id_list = embedding_tuple
    distance_matrix, index_matrix = get_distance_tuple(
        embedding_matrix=embedding_matrix,
        distance_metric=distance_metric)
    true_matches_dict = get_true_matches_dict(
        df=eval_df,
        progress_bar=True)
    phash_pred_matches_dict = get_phash_pred_matches_dict(df=eval_df, progress_bar=True) if use_phash else None

    if isinstance(threshold, ThresholdFinder):
        threshold_finder = threshold
        threshold = threshold_finder.threshold
        pred_matches_dict = {}
        while threshold is not None:
            distance_pred_matches_dict = get_distance_pred_matches_dict(
                distance_matrix=distance_matrix,
                index_matrix=index_matrix,
                posting_id_list=posting_id_list,
                threshold=threshold,
                progress_bar=True)
            pred_matches_dict = join_matches_dict_list([
                distance_pred_matches_dict,
                phash_pred_matches_dict,
            ]) if phash_pred_matches_dict is not None else distance_pred_matches_dict
            print(len(pred_matches_dict), len(eval_df))
            mean_num_matches = mean([len(m_set) for m_set in pred_matches_dict.values()])
            print(f'threshold = {threshold}, mean_num_matches = {mean_num_matches}')
            threshold = threshold_finder.get_next_threshold(mean_num_matches)
    elif isinstance(threshold, float):
        distance_pred_matches_dict = get_distance_pred_matches_dict(
            distance_matrix=distance_matrix,
            index_matrix=index_matrix,
            posting_id_list=posting_id_list,
            threshold=threshold,
            progress_bar=True)
        pred_matches_dict = join_matches_dict_list([
            distance_pred_matches_dict,
            phash_pred_matches_dict,
        ]) if phash_pred_matches_dict is not None else distance_pred_matches_dict
    else:
        raise TypeError(f'Invalid threshold type: {type(threshold)}.')

    if transitive_relations_rounds > 0:
        pred_matches_dict = build_transitive_relations(pred_matches_dict, rounds=transitive_relations_rounds)
    if postprocess:
        pred_matches_dict = postprocess_matches_dict(pred_matches_dict)

    score = get_f1_mean_for_matches(
        true_matches_dict=true_matches_dict,
        pred_matches_dict=pred_matches_dict)
    print(f'threshold = {threshold}, score = {score}')


def get_matches_dict_clustering(
        embedding_tuple: Tuple[np.ndarray, List[str]],
        threshold: Union[ThresholdFinder, float]):
    embedding_matrix, posting_id_list = embedding_tuple
    ac = AgglomerativeClustering(distance_threshold=threshold, n_clusters=None, linkage='single')
    labels = ac.fit_predict(embedding_matrix)

    pred_matches_dict = {}
    posting_id_series = pd.Series(posting_id_list)
    for pid, label in zip(posting_id_list, labels.tolist()):
        pred_matches_dict[pid] = set(posting_id_series[labels == label].tolist())

    return pred_matches_dict


def evaluate_embeddings_clustering(
        embedding_tuple: Tuple[np.ndarray, List[str]],
        index_root_path: str,
        threshold: Union[ThresholdFinder, float],
        distance_metric: str = 'euclidean',
        test_set_file_name: str = 'test-set.csv'):
    eval_df = pd.read_csv(Path(index_root_path) / test_set_file_name)

    embedding_matrix, posting_id_list = embedding_tuple
    ac = AgglomerativeClustering(
        distance_threshold=threshold, n_clusters=None, linkage='single', affinity=distance_metric)
    labels = ac.fit_predict(embedding_matrix)

    pred_matches_dict = {}
    posting_id_series = pd.Series(posting_id_list)
    for pid, label in zip(posting_id_list, labels.tolist()):
        pred_matches_dict[pid] = set(posting_id_series[labels == label].tolist())

    true_matches_dict = get_true_matches_dict(
        df=eval_df,
        progress_bar=True)
    score = get_f1_mean_for_matches(
        true_matches_dict=true_matches_dict,
        pred_matches_dict=pred_matches_dict)
    print(f'threshold = {threshold}, score = {score}')
