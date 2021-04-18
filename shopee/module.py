from typing import Union, Dict, List, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import LightningLoggerBase
from typing_extensions import TypedDict


class StepResult(TypedDict):
    loss: torch.Tensor
    num_correct: int
    num_total: int


class Module(LightningModule):
    logger: Optional[LightningLoggerBase]

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().__call__(x, *args, **kwargs)

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        return items

    def training_epoch_end(self, outputs: List[StepResult]):
        loss: float = torch.stack([r['loss'] for r in outputs]).mean().item()
        accuracy = sum([r['num_correct'] for r in outputs]) / sum([r['num_total'] for r in outputs])
        self.log(name='train_loss', value=loss, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        self.log(name='train_accuracy', value=accuracy, prog_bar=True, on_epoch=True, on_step=False, logger=True)

    def validation_epoch_end(self, outputs: List[StepResult]):
        loss: float = torch.stack([r['loss'] for r in outputs]).mean().item()
        accuracy = sum([r['num_correct'] for r in outputs]) / sum([r['num_total'] for r in outputs])
        self.log(name='valid_loss', value=loss, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        self.log(name='valid_accuracy', value=accuracy, prog_bar=True, on_epoch=True, on_step=False, logger=True)
