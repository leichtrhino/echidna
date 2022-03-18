
import typing as tp
import inspect
import random
import numpy
import torch

from .nest import (
    BaselineEncDecModel,
    BaselineChimeraNet,
)
from . import checkpoints

_model_classes = {}
def register_model_class(name : str,
                         klass : tp.Callable):
    _model_classes[name] = klass
def get_model_class(name : str):
    if name not in _model_classes:
        raise ValueError(f'{name} is not registered as a model class')
    return _model_classes[name]

if len(_model_classes) == 0:
    register_model_class('baseline_encdec', BaselineEncDecModel)
    register_model_class('baseline_chimera', BaselineChimeraNet)


class Model(object):
    def get_class(self) -> str:
        """
        get model_class
        """
        raise NotImplementedError()

    def get_hyperparameters(self) -> dict:
        """
        get model hyperparaemters to give model_class
        """
        raise NotImplementedError()

    def get_torch_model(self) -> torch.nn.Module:
        """
        get parameters of its model
        """
        raise NotImplementedError()

    def get_epoch(self) -> int:
        """
        get epoch of model
        """
        return 0

    def save_torch_model(self, path):
        """
        Save model
        """

        # move model to cpu
        device = next(self.get_torch_model().parameters()).device
        self.get_torch_model().to('cpu')

        # save
        model_dict = {
            'class': self.get_class(),
            'hyperparameters': self.get_hyperparameters(),
            'epoch': self.get_epoch(),
            'state': self.get_torch_model().state_dict(),
        }
        torch.save(model_dict, path)

        # move model to device
        self.get_torch_model().to(device)

    def to_dict(self):
        return {
            'type': _reverse_model_interface_type[type(self)],
            'args': self.to_dict_args(),
        }

    @classmethod
    def from_dict(cls, d : dict):
        if_type = d['type']
        if_class = _model_interface_type[if_type]
        return if_class.from_dict_args(d['args'])

    def to_dict_args(self):
        raise NotImplementedError()

class InitialModel(Model):
    def __init__(self,
                 klass : str,
                 hyperparameters : dict,
                 seed : int,
                 ):

        self.klass = klass
        self.hyperparameters = hyperparameters
        self.seed = seed
        self.torch_model = None

    def get_class(self) -> str:
        return self.klass

    def get_hyperparameters(self) -> dict:
        return self.hyperparameters

    def get_torch_model(self):
        if self.torch_model is None:
            random.seed(self.seed)
            numpy.random.seed(self.seed)
            torch.manual_seed(self.seed)
            self.torch_model = get_model_class \
                (self.klass)(self.hyperparameters)

        return self.torch_model

    def to_dict_args(self):
        return {
            'class': self.klass,
            'hyperparameters': self.hyperparameters,
            'seed': self.seed,
        }

    @classmethod
    def from_dict_args(cls, d : dict):
        return cls(
            klass=d['class'],
            hyperparameters=d['hyperparameters'],
            seed=d.get('seed'),
        )

class SavedModel(Model):
    def __init__(self,
                 path : str):
        self.path = path
        self.klass = None
        self.hyperparameters = None
        self.epoch = None
        self.torch_model = None

    def _load(self):
        model_dict = torch.load(self.path)
        self.klass = model_dict['class']
        self.hyperparameters = model_dict['hyperparameters']
        self.torch_model = get_model_class \
                (self.klass)(self.hyperparameters)
        self.epoch = model_dict['epoch']
        self.torch_model.load_state_dict(model_dict['state'])

    def get_class(self):
        if self.klass is None:
            self._load()
        return self.klass

    def get_hyperparameters(self):
        if self.hyperparameters is None:
            self._load()
        return self.hyperparameters

    def get_epoch(self):
        if self.epoch is None:
            self._load()
        return self.epoch

    def get_torch_model(self):
        if self.torch_model is None:
            self._load()
        return self.torch_model

    def to_dict_args(self):
        return {
            'path': self.path,
        }

    @classmethod
    def from_dict_args(cls, d : dict):
        return cls(
            path=d['path'],
        )

class CheckpointModel(Model):
    def __init__(self,
                 checkpoint):
        self.checkpoint = checkpoint
        self.model = None

    def get_class(self) -> str:
        """
        get model_class
        """
        if self.model is None:
            self.model = self.checkpoint.get_model()
        return self.model.get_class()

    def get_hyperparameters(self) -> dict:
        """
        get model hyperparaemters to give model_class
        """
        if self.model is None:
            self.model = self.checkpoint.get_model()
        return self.model.get_hyperparameters()

    def get_epoch(self) -> int:
        """
        get epoch of the model
        """
        return self.checkpoint.get_torch_scheduler().last_epoch

    def get_torch_model(self) -> torch.nn.Module:
        """
        get parameters of its model
        """
        if self.model is None:
            self.model = self.checkpoint.get_model()
        return self.model.get_torch_model()


    def to_dict_args(self):
        return {
            'checkpoint': self.checkpoint.to_dict(),
        }

    @classmethod
    def from_dict_args(cls, d : dict):
        return cls(
            checkpoint=checkpoints.Checkpoint.from_dict(d['checkpoint']),
        )

_model_interface_type = {
    'initial': InitialModel,
    'saved': SavedModel,
    'checkpoint': CheckpointModel,
}
_reverse_model_interface_type = dict(
    (v, k) for k, v in _model_interface_type.items())

