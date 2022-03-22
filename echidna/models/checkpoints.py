
import typing as tp
import inspect
import random
import numpy
import torch

from .nest import (
    BaselineEncDecModel,
    BaselineChimeraNet,
)
from . import models

_optimizer_class = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
}
def get_optimizer_class(name : str):
    if name not in _optimizer_class:
        raise ValueError(f'{name} is not valid optimizer class')
    return _optimizer_class[name]

_scheduler_class = {
    'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'cyclic': torch.optim.lr_scheduler.CyclicLR,
}
def get_scheduler_class(name : str):
    if name not in _scheduler_class:
        raise ValueError(f'{name} is not valid scheduler class')
    return _scheduler_class[name]

class Checkpoint(object):
    def get_model(self):
        raise NotImplementedError()

    def get_torch_model(self):
        return self.get_model().get_torch_model()

    def get_model_class(self):
        return self.get_model().get_class()

    def get_model_hyperparameters(self):
        return self.get_model().get_hyperparameters()

    def get_epoch(self):
        return self.get_torch_scheduler().last_epoch

    def get_torch_optimizer(self):
        raise NotImplementedError()

    def get_optimizer_class(self):
        raise NotImplementedError()

    def get_optimizer_hyperparameters(self):
        raise NotImplementedError()

    def get_torch_scheduler(self):
        raise NotImplementedError()

    def get_scheduler_class(self):
        raise NotImplementedError()

    def get_scheduler_hyperparameters(self):
        raise NotImplementedError()


    def save_torch_checkpoint(self, path):
        """
        Save checkpoint
        """

        # move everything to cpu
        device = next(self.get_torch_model().parameters()).device
        self.get_torch_model().to('cpu')
        for stae in self.get_torch_optimizer().state.values():
            for k, v in state.items():
                if type(v) == torch.Tensor:
                    state[k] = v.to('cpu')

        # save
        checkpoint_dict = {
            'class': self.get_model().get_class(),
            'hyperparameters': self.get_model().get_hyperparameters(),
            'epoch': self.get_torch_scheduler().last_epoch,
            'state': self.get_model().get_torch_model().state_dict(),
            'optimizer': {
                'class': self.get_optimizer_class(),
                'hyperparameters': self.get_optimizer_hyperparameters(),
                'state': self.get_torch_optimizer().state_dict(),
            },
            'scheduler': {
                'class': self.get_scheduler_class(),
                'hyperparameters': self.get_scheduler_hyperparameters(),
                'state': self.get_torch_scheduler().state_dict(),
            },
        }
        torch.save(checkpoint_dict, path)

        # move everything to device
        self.get_torch_model().to(device)
        for state in self.get_torch_optimizer().state.values():
            for k, v in state.items():
                if type(v) == torch.Tensor:
                    state[k] = v.to(device)

    def to_dict(self):
        return {
            'type': _reverse_checkpoint_interface_type[type(self)],
            'args': self.to_dict_args(),
        }

    @classmethod
    def from_dict(cls, d : dict):
        if_type = d['type']
        if_class = _checkpoint_interface_type[if_type]
        return if_class.from_dict_args(d['args'])

    def to_dict_args(self):
        raise NotImplementedError()

class InitialCheckpoint(Checkpoint):
    def __init__(self,
                 model,
                 optimizer_class : str,
                 optimizer_args : dict,
                 scheduler_class : str,
                 scheduler_args : dict,
                 seed : int):
        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_args = optimizer_args
        self.optimizer = None
        self.scheduler_class = scheduler_class
        self.scheduler_args = scheduler_args
        self.scheduler = None
        self.seed = seed

    def get_model(self):
        return self.model

    def get_torch_optimizer(self):
        if self.optimizer is None:
            base_lr = self.optimizer_args.get('lr')
            parameters = self.get_torch_model().parameter_list(base_lr)
            random.seed(self.seed)
            numpy.random.seed(self.seed)
            torch.manual_seed(self.seed)
            self.optimizer = _optimizer_class[self.optimizer_class](
                parameters, **self.optimizer_args)

        return self.optimizer

    def get_optimizer_class(self):
        return self.optimizer_class

    def get_optimizer_hyperparameters(self):
        return self.optimizer_args

    def get_torch_scheduler(self):
        if self.scheduler is None:
            random.seed(self.seed)
            numpy.random.seed(self.seed)
            torch.manual_seed(self.seed)
            self.scheduler = _scheduler_class[self.scheduler_class](
                self.get_torch_optimizer(), **self.scheduler_args)

        return self.scheduler

    def get_scheduler_class(self):
        return self.scheduler_class

    def get_scheduler_hyperparameters(self):
        return self.scheduler_args


    def to_dict_args(self):
        return {
            'model': self.model.to_dict(),
            'optimizer_class': self.optimizer_class,
            'optimizer_args': self.optimizer_args,
            'scheduler_class': self.scheduler_class,
            'scheduler_args': self.scheduler_args,
            'seed': self.seed,
        }

    @classmethod
    def from_dict_args(cls, d : dict):
        return cls(
            model=models.Model.from_dict(d['model']),
            optimizer_class=d['optimizer_class'],
            optimizer_args=d['optimizer_args'],
            scheduler_class=d['scheduler_class'],
            scheduler_args=d['scheduler_args'],
            seed=d.get('seed')
        )

class SavedCheckpoint(Checkpoint):
    def __init__(self,
                 path : str):
        self.path = path
        self.model = None
        self.optimizer_class = None
        self.optimizer_args = None
        self.torch_optimizer = None
        self.scheduler_class = None
        self.scheduler_args = None
        self.torch_scheduler = None

    def _load(self):
        checkpoint_dict = torch.load(self.path)
        # load model
        self.model = models.SavedModel(self.path)
        self.model.klass = checkpoint_dict['class']
        self.model.hyperparameters = checkpoint_dict['hyperparameters']
        self.model.torch_model = models.get_model_class \
                (self.model.klass)(self.model.hyperparameters)
        self.model.epoch = checkpoint_dict['epoch']
        self.model.torch_model.load_state_dict(checkpoint_dict['state'])
        # load optimizer
        self.optimizer_class = checkpoint_dict['optimizer']['class']
        self.optimizer_args = checkpoint_dict['optimizer']['hyperparameters']
        base_lr = self.optimizer_args.get('lr')
        self.torch_optimizer = get_optimizer_class \
            (self.optimizer_class)(
                self.model.torch_model.parameter_list(base_lr),
                **self.optimizer_args
            )
        self.torch_optimizer.load_state_dict(
            checkpoint_dict['optimizer']['state'])
        # load scheduler
        self.scheduler_class = checkpoint_dict['scheduler']['class']
        self.scheduler_args = checkpoint_dict['scheduler']['hyperparameters']
        self.torch_scheduler = get_scheduler_class \
            (self.scheduler_class) \
            (self.torch_optimizer, **self.scheduler_args)
        self.torch_scheduler.load_state_dict(
            checkpoint_dict['scheduler']['state'])

    def get_model(self):
        if self.model is None:
            self._load()

        return self.model

    def get_torch_optimizer(self):
        if self.torch_optimizer is None:
            self._load()

        return self.torch_optimizer

    def get_optimizer_class(self):
        if self.optimizer_class is None:
            self._load()

        return self.optimizer_class

    def get_optimizer_hyperparameters(self):
        if self.optimizer_args is None:
            self._load()

        return self.optimizer_args

    def get_torch_scheduler(self):
        if self.torch_scheduler is None:
            self._load()

        return self.torch_scheduler

    def get_scheduler_class(self):
        if self.scheduler_class is None:
            self._load()

        return self.scheduler_class

    def get_scheduler_hyperparameters(self):
        if self.scheduler_args is None:
            self._load()

        return self.scheduler_args

    def to_dict_args(self):
        return {
            'path': self.path,
        }

    @classmethod
    def from_dict_args(cls, d : dict):
        return cls(
            path=d['path'],
        )

_checkpoint_interface_type = {
    'initial': InitialCheckpoint,
    'saved': SavedCheckpoint,
}
_reverse_checkpoint_interface_type = dict(
    (v, k) for k, v in _checkpoint_interface_type.items())

