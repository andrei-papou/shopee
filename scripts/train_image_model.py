from pathlib import Path

import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.checkpoint import save_checkpoint
from lib.datasets import PrecomputedTripletImageDataset, RandomTripletImageDataset
from lib.models import ResNet18


def main():
    lr = 1e-4
    num_epochs = 25
    margin = 1.0
    max_epochs_no_improvement = 7
    train_batch_size = 16
    test_batch_size = 16
    num_workers = 8
    checkpoint_path = '/home/andrei/projects/shopee/models/resnet18-triplet-random-checkpoint-1.pth'
    pretrained_model_path = '/home/andrei/projects/shopee/models/resnet18-triplet-random-3.pth'

    tb_writer = SummaryWriter()

    data_root_path = Path('/media/hdd/home/andrew/kaggle-data/shopee-product-matching')
    image_folder_path = data_root_path / 'train_images'

    train_df = pd.read_csv(data_root_path / 'train-set.csv')
    test_pair_df = pd.read_csv(data_root_path / 'test_triplets.csv')

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
    model.load_state_dict(torch.load(pretrained_model_path))
    loss_fn = torch.nn.TripletMarginLoss(margin=margin)
    opt = Adam(model.parameters(), lr=lr)
    lr_scheduler = ReduceLROnPlateau(optimizer=opt, mode='max', factor=0.1, patience=3)

    best_accuracy = 0.0
    epochs_no_improvement = 0

    for ep in range(num_epochs):
        model.train(True)

        train_num_correct = 0
        train_loss = 0.0
        for ax, px, nx in tqdm(train_data_loader, total=len(train_data_loader)):
            opt.zero_grad()
            ax, px, nx = ax.cuda(), px.cuda(), nx.cuda()
            v = model(torch.cat([ax, px, nx], dim=0))
            split_size = int(v.shape[0] / 3)
            av, pv, nv = v[:split_size], v[split_size:split_size * 2], v[split_size * 2:]
            loss = loss_fn(av, pv, nv)
            loss.backward()
            opt.step()
            with torch.no_grad():
                av = torch.unsqueeze(av, dim=1)
                pv = torch.unsqueeze(pv, dim=1)
                nv = torch.unsqueeze(nv, dim=1)
                train_num_correct += (torch.cdist(av, nv) - torch.cdist(av, pv) > margin).int().sum()
                train_loss += loss.sum().item()
        train_loss /= len(train_dataset)
        train_accuracy = train_num_correct / len(train_dataset)

        model.eval()
        test_num_correct = 0
        test_loss = 0.0
        with torch.no_grad():
            for ax, px, nx in tqdm(test_data_loader, total=len(test_data_loader)):
                ax, px, nx = ax.cuda(), px.cuda(), nx.cuda()
                v = model(torch.cat([ax, px, nx], dim=0))
                split_size = int(v.shape[0] / 3)
                av, pv, nv = v[:split_size], v[split_size:split_size * 2], v[split_size * 2:]
                loss = loss_fn(av, pv, nv)
                av = torch.unsqueeze(av, dim=1)
                pv = torch.unsqueeze(pv, dim=1)
                nv = torch.unsqueeze(nv, dim=1)
                test_num_correct += (torch.cdist(av, nv) - torch.cdist(av, pv) > margin).int().sum()
                test_loss += loss.sum().item()
        test_accuracy = test_num_correct / len(test_dataset)

        lr_scheduler.step(test_accuracy)

        print(f'epoch: {ep}, train_accuracy: {train_accuracy}, test_accuracy: {test_accuracy}')
        tb_writer.add_scalar(tag='train_loss', scalar_value=train_loss, global_step=ep)
        tb_writer.add_scalars(main_tag='accuracy', tag_scalar_dict={
            'train': train_accuracy,
            'test': test_accuracy,
        }, global_step=ep)

        if test_accuracy > best_accuracy:
            print(f'Best accuracy improved: {test_accuracy} vs {best_accuracy}. Saving the model.')
            save_checkpoint(checkpoint={
                'model_state_dict': model.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'lr_scheduler': lr_scheduler,
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


if __name__ == '__main__':
    main()
