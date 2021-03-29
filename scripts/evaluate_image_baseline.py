from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.datasets import ImageTestPairDataset
from lib.models import ResNet18


def main():
    threshold = 21.0
    batch_size = 32
    data_root = Path('/media/hdd/home/andrew/kaggle-data/shopee-product-matching')

    train_df = pd.read_csv(data_root / 'test_pairs.csv')

    model = ResNet18()
    model = model.cuda()
    model.eval()

    num_correct = 0.0
    dataset = ImageTestPairDataset(df=train_df, image_folder_path=data_root / 'train_images')
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
    with torch.no_grad():
        for x1, x2, y in tqdm(data_loader, total=len(data_loader)):
            x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
            v = model(torch.cat([x1, x2], dim=0))
            split_idx = int(v.shape[0] / 2)
            v1, v2 = v[:split_idx, ...], v[split_idx:, ...]
            dist = torch.cdist(torch.unsqueeze(v1, dim=1), torch.unsqueeze(v2, dim=1)).squeeze()
            y_pred = (dist < threshold).int()
            y = y.squeeze()
            num_correct += (y == y_pred).float().sum().item()

    print(f'accuracy = {num_correct / len(dataset)}')


if __name__ == '__main__':
    main()
