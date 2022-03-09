
import typing as tp
import inspect
from datetime import datetime
import logging
import json
import random
import torch

from ..data.datasets import Dataset
from ..models.checkpoints import Checkpoint
from ..metrics.loss import Loss
from . import utils

class TrainingSpec(object):
    def __init__(self,
                 # checkpoint
                 checkpoint : Checkpoint,

                 # dataset params
                 training_dataset : Dataset,
                 training_sample_size : int,
                 validation_dataset : tp.Optional[Dataset],
                 validation_sample_size : int,

                 # output params
                 checkpoint_pattern,
                 journal_pattern,
                 log_pattern,
                 log_level,

                 # misc. training param
                 loss_function : Loss,
                 batch_size=32,
                 compute_batch_size=None,
                 training_epochs=1,
                 max_grad_norm=None,
                 n_fft : int=2048,
                 hop_length : int=512,
                 seed=None,
                 device='cpu',
                 jobs=0,
                 ):

        # checkpoint
        self.checkpoint = checkpoint

        # dataset params
        self.training_dataset = training_dataset
        self.training_sample_size = training_sample_size
        self.validation_dataset = validation_dataset
        self.validation_sample_size = validation_sample_size

        # output params
        self.checkpoint_pattern = str(checkpoint_pattern)
        self.journal_pattern = str(journal_pattern)
        self.log_pattern = str(log_pattern)
        self.log_level = log_level

        # misc. training param
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.compute_batch_size = compute_batch_size
        self.training_epochs = training_epochs
        self.max_grad_norm = max_grad_norm
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.seed = seed
        self.device = device
        self.jobs = jobs

        # check checkpoint_pattern, journal_pattern, log_pattern
        self._validate_patterns()


    def to_dict(self):
        return {
            # checkpoint
            'checkpoint': self.checkpoint.to_dict(),
            # dataset params
            'training_dataset': self.training_dataset.to_dict(),
            'training_sample_size': self.training_sample_size,
            'validation_dataset': self.validation_dataset.to_dict()
            if self.validation_dataset is not None else None,
            'validation_sample_size': self.validation_sample_size,
            # output params
            'checkpoint_pattern': self.checkpoint_pattern,
            'journal_pattern': self.journal_pattern,
            'log_pattern': self.log_pattern,
            'log_level': self.log_level,
            # misc. training param
            'loss_function': self.loss_function.to_dict(),
            'batch_size': self.batch_size,
            'compute_batch_size': self.compute_batch_size,
            'training_epochs': self.training_epochs,
            'max_grad_norm': self.max_grad_norm,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'seed': self.seed,
            'device': self.device,
            'jobs': self.jobs,
        }

    @classmethod
    def from_dict(cls, d : dict):
        return cls(
            # checkpoint
            checkpoint=Checkpoint.from_dict(d['checkpoint']),
            # dataset params
            training_dataset=Dataset.from_dict(d['training_dataset']),
            training_sample_size=d['training_sample_size'],
            validation_dataset=Dataset.from_dict(d['validation_dataset'])
            if d.get('validation_dataset', None) is not None else None,
            validation_sample_size=d.get('validation_sample_size'),
            # output params
            checkpoint_pattern=d['checkpoint_pattern'],
            journal_pattern=d['journal_pattern'],
            log_pattern=d['log_pattern'],
            log_level=d['log_level'],
            # misc. training param
            loss_function=Metrics.from_dict(d['loss_function']),
            batch_size=d.get('batch_size'),
            compute_batch_size=d.get('compute_batch_size'),
            training_epochs=d.get('training_epochs', 1),
            max_grad_norm=d.get('max_grad_norm'),
            n_fft=d.get('n_fft', 2048),
            hop_length=d.get('hop_length', 512),
            seed=d.get('seed'),
            device=d.get('device', 'cpu'),
            jobs=d.get('jobs', 0),
        )

    def _validate_patterns(self):
        template_before_epoch = {
            # from specification
            'model': self.checkpoint.get_model_class(),
            'model_epoch': self.checkpoint.get_epoch(),
            'optimizer': self.checkpoint.get_optimizer_class(),
            'lr': self.checkpoint.get_torch_optimizer().defaults['lr'],
            'scheduler': self.checkpoint.get_scheduler_class(),
            # from epoch journal
            'process_start': datetime.now(),
            'training_epoch': 0,
        }
        template_after_epoch = {
            # from specification
            'model': self.checkpoint.get_model_class(),
            'model_epoch': self.checkpoint.get_epoch(),
            'optimizer': self.checkpoint.get_optimizer_class(),
            'lr': self.checkpoint.get_torch_optimizer().defaults['lr'],
            'scheduler': self.checkpoint.get_scheduler_class(),
            # from epoch journal
            'process_start': datetime.now(),
            'process_end': datetime.now(),
            'training_epoch': 0,
            'training_loss': 0.0,
            'validation_loss': 0.0,
        }

        if self.log_pattern:
            self.log_pattern.format(**template_before_epoch)
        if self.checkpoint_pattern:
            self.checkpoint_pattern.format(**template_after_epoch)
        if self.journal_pattern:
            self.journal_pattern.format(**template_after_epoch)

    def train(self):
        _train(self)


