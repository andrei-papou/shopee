import os

from pytorch_lightning.callbacks import ModelCheckpoint


def create_checkpoint_callback(
        checkpoint_file_path: str,
        monitor: str = 'valid_accuracy',
        mode: str = 'max') -> ModelCheckpoint:
    checkpoint_dir_path = os.path.dirname(checkpoint_file_path)
    checkpoint_file_name = os.path.basename(checkpoint_file_path)
    return ModelCheckpoint(
        dirpath=checkpoint_dir_path,
        filename=f'{checkpoint_file_name}-{{{monitor}:.4f}}',
        monitor=monitor,
        mode=mode)
