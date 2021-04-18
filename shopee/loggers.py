import argparse
import sys
from typing import Union, Dict, Optional, Any, TextIO

from pytorch_lightning.loggers import LightningLoggerBase


def _dict_to_str(d: Dict[str, float]) -> str:
    return ', '.join([f'{mk} = {mv}' for (mk, mv) in d.items()])


class TextIOWriter:

    def __init__(self, io: TextIO, prefix: str = ''):
        self._io = io
        self._prefix = prefix

    def write_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        self._io.flush()
        self._io.write(f'\n[metrics] {self._prefix}: step = {step}, {_dict_to_str(metrics)}\n')
        self._io.flush()

    def write_hparams(self, params: Dict[str, float]):
        self._io.flush()
        self._io.write(f'\n[hparams] {self._prefix}: {_dict_to_str(params)}\n')
        self._io.flush()


class TextIOLogger(LightningLoggerBase):

    def __init__(
            self,
            io: TextIO = sys.stdout,
            name: str = 'default',
            version: Optional[Union[int, str]] = None,):
        super().__init__()
        self._name = name
        if version is None:
            version = 'version_0'
        elif isinstance(version, int):
            version = f'version_{version}'
        elif isinstance(version, str):
            version = version
        else:
            raise TypeError(f'Unexpected type of version: {type(version)}.')
        self._version: str = version
        self._experiment = TextIOWriter(io=io, prefix=f'{self._name}/{self._version}')

    @property
    def experiment(self) -> TextIOWriter:
        return self._experiment

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        self._experiment.write_metrics(metrics=metrics, step=step)

    def log_hyperparams(self, params: Union[Dict[str, Any], argparse.Namespace], *args, **kwargs):
        params = self._convert_params(params)
        self._experiment.write_hparams(params=params)

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> Union[int, str]:
        return self._version
