from shopee.contrastive import train_model


def main():
    train_model(
        index_root_path='/media/hdd/home/andrew/kaggle-data/shopee-product-matching',
        data_root_path='/media/hdd/home/andrew/kaggle-data/shopee-product-matching',
        checkpoint_path='/home/andrei/projects/shopee/models/resnet18-contrastive-checkpoint-1.pth',
        train_batch_size=16,
        test_batch_size=16,
        num_workers=8)


if __name__ == '__main__':
    main()
