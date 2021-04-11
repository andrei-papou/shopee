from pathlib import Path
from typing import Tuple, Callable, Optional

import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from shopee.checkpoint import save_checkpoint, load_checkpoint
from shopee.datasets import RandomTripletImageDataset, PrecomputedTripletImageDataset
from shopee.evaluation import get_embedding_dict, get_true_matches_dict, get_pred_matches_dict, get_f1_mean_for_matches
from shopee.loss import ContrastiveLoss
from shopee.models import ResNet18


def get_triplet_embedding(
        model: torch.nn.Module,
        ax: torch.Tensor,
        px: torch.Tensor,
        nx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v = model(torch.cat([ax, px, nx], dim=0))
    split_size = int(v.shape[0] / 3)
    return v[:split_size], v[split_size:split_size * 2], v[split_size * 2:]


def get_loss(
        loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        av: torch.Tensor,
        pv: torch.Tensor,
        nv: torch.Tensor) -> torch.Tensor:
    n = av.shape[0]
    pos_loss = loss_fn(av, pv, torch.ones(n).cuda())
    neg_loss = loss_fn(av, nv, torch.zeros(n).cuda())
    return (pos_loss + neg_loss) / 2


def get_num_correct(
        av: torch.Tensor,
        pv: torch.Tensor,
        nv: torch.Tensor,
        margin: float) -> int:
    av = torch.unsqueeze(av, dim=1)
    pv = torch.unsqueeze(pv, dim=1)
    nv = torch.unsqueeze(nv, dim=1)
    return (torch.cdist(av, pv) <= margin).int().sum().item() + \
           (torch.cdist(av, nv) > margin).int().sum().item()


def train_model(
        index_root_path: str,
        data_root_path: str,
        checkpoint_path: str,
        lr: float = 1e-3,
        num_epochs: int = 25,
        margin: float = 20.0,
        eval_margin: float = 10.0,
        max_epochs_no_improvement: int = 7,
        train_batch_size: int = 64,
        test_batch_size: int = 64,
        num_workers: int = 2,
        start_from_checkpoint_path: Optional[str] = None):
    data_root_path = Path(data_root_path)
    image_folder_path = data_root_path / 'train_images'

    index_root_path = Path(index_root_path)
    train_df = pd.read_csv(index_root_path / 'train-set.csv')
    test_pair_df = pd.read_csv(index_root_path / 'test_triplets.csv')
    eval_df = pd.read_csv(index_root_path / 'test-set.csv')

    train_dataset = RandomTripletImageDataset(df=train_df, image_folder_path=image_folder_path)
    test_dataset = PrecomputedTripletImageDataset(df=test_pair_df, image_folder_path=image_folder_path)
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True)
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        num_workers=num_workers,
        pin_memory=True)

    model = ResNet18().cuda()
    loss_fn = ContrastiveLoss(margin=margin)
    opt = Adam(model.parameters(), lr=lr)
    lr_scheduler = ReduceLROnPlateau(optimizer=opt, mode='max', factor=0.1, patience=3)
    if start_from_checkpoint_path is not None:
        checkpoint_dict = load_checkpoint(start_from_checkpoint_path)
        model.load_state_dict(checkpoint_dict['model_state_dict'])
        opt.load_state_dict(checkpoint_dict['opt_state_dict'])
        lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler_state_dict'])

    best_accuracy = 0.0
    epochs_no_improvement = 0

    for ep in range(num_epochs):
        model.train(True)

        train_num_correct = 0
        train_loss = 0.0
        for ax, px, nx in tqdm(train_data_loader, total=len(train_data_loader), desc='training'):
            opt.zero_grad()
            ax, px, nx = ax.cuda(), px.cuda(), nx.cuda()
            av, pv, nv = get_triplet_embedding(model=model, ax=ax, px=px, nx=nx)
            loss = get_loss(loss_fn=loss_fn, av=av, pv=pv, nv=nv)
            loss.backward()
            opt.step()
            with torch.no_grad():
                train_num_correct += get_num_correct(av=av, pv=pv, nv=nv, margin=margin)
                train_loss += loss.sum().item()
        train_loss /= len(train_dataset)
        train_accuracy = train_num_correct / (len(train_dataset) * 2)

        model.eval()
        test_num_correct = 0
        test_loss = 0.0
        with torch.no_grad():
            for ax, px, nx in tqdm(test_data_loader, total=len(test_data_loader), desc='validation'):
                ax, px, nx = ax.cuda(), px.cuda(), nx.cuda()
                av, pv, nv = get_triplet_embedding(model=model, ax=ax, px=px, nx=nx)
                loss = get_loss(loss_fn=loss_fn, av=av, pv=pv, nv=nv)
                test_num_correct += get_num_correct(av=av, pv=pv, nv=nv, margin=margin)
                test_loss += loss.sum().item()
        test_accuracy = test_num_correct / (len(test_dataset) * 2)

        with torch.no_grad():
            embedding_dict = get_embedding_dict(df=eval_df, model=model, data_path=image_folder_path, progress_bar=True)
            true_matches_dict = get_true_matches_dict(df=eval_df, progress_bar=True)
            pred_matches_dict = get_pred_matches_dict(
                embedding_dict=embedding_dict, threshold=eval_margin, progress_bar=True)
            f1_mean = get_f1_mean_for_matches(true_matches_dict=true_matches_dict, pred_matches_dict=pred_matches_dict)

        lr_scheduler.step(test_accuracy)

        print(f'epoch: {ep}, train_accuracy: {train_accuracy}, test_accuracy: {test_accuracy}, f1_mean: {f1_mean}')

        if test_accuracy > best_accuracy:
            print(f'Best accuracy improved: {test_accuracy} vs {best_accuracy}. Saving the model.')
            save_checkpoint(checkpoint={
                'model_state_dict': model.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            }, path=checkpoint_path)
            best_accuracy = test_accuracy
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            print(f'No improvement in {epochs_no_improvement}/{max_epochs_no_improvement} epochs. '
                  f'Best accuracy: {best_accuracy}.')

        if epochs_no_improvement >= max_epochs_no_improvement:
            print(f'No improvement in {max_epochs_no_improvement} epochs. Stopping the training.')
            break
