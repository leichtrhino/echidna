
from datetime import datetime
import logging
import json
import random
import torch

from ..data.datasets import Dataset
from ..models.models import Model
from ..metrics import Loss
from . import utils

class ValidationSpec(object):
    def __init__(self,
                 # model
                 model : Model,

                 # dataset params
                 validation_dataset : Dataset,
                 validation_sample_size : int,

                 # output params
                 journal_pattern,
                 log_pattern,
                 log_level,

                 # misc. training param
                 loss_function : Loss,
                 batch_size=32,
                 compute_batch_size=None,
                 n_fft : int=2048,
                 hop_length : int=512,
                 device='cpu',
                 jobs=0,
                 ):

        # model
        self.model = model

        # dataset params
        self.validation_dataset = validation_dataset
        self.validation_sample_size = validation_sample_size

        # output params
        self.journal_pattern = str(journal_pattern)
        self.log_pattern = str(log_pattern)
        self.log_level = log_level

        # misc. training param
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.compute_batch_size = compute_batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        self.jobs = jobs

        # check journal_pattern and log_pattern
        self._validate_patterns()

    def to_dict(self):
        return {
            # model
            'model': self.model.to_dict(),
            # dataset params
            'validation_dataset': self.validation_dataset.to_dict(),
            'validation_sample_size': self.validation_sample_size,
            # output params
            'journal_pattern': self.journal_pattern,
            'log_pattern': self.log_pattern,
            'log_level': self.log_level,
            # misc. validation param
            'loss_function': self.loss_function.to_dict(),
            'batch_size': self.batch_size,
            'compute_batch_size': self.compute_batch_size,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'device': self.device,
        }

    @classmethod
    def from_dict(cls, d : dict):
        return cls(
            # model
            model=Model.from_dict(d['model']),
            # dataset params
            validation_dataset=Dataset.from_dict(d['validation_dataset']),
            validation_sample_size=d.get('validation_sample_size'),
            # output params
            journal_pattern=d['journal_pattern'],
            log_pattern=d['log_pattern'],
            log_level=d['log_level'],
            # misc. training param
            loss_function=Metrics.from_dict(d['loss_function']),
            batch_size=d.get('batch_size'),
            compute_batch_size=d.get('compute_batch_size'),
            n_fft=d.get('n_fft', 2048),
            hop_length=d.get('hop_length', 512),
            device=d.get('device', 'cpu'),
        )

    def _validate_patterns(self):
        template_before_process = {
            # from specification
            'model': self.model.get_class(),
            'model_epoch': self.model.get_epoch(),
            # from validation journal
            'process_start': datetime.now(),
        }
        template_after_process = {
            # from specification
            'model': self.model.get_class(),
            'model_epoch': self.model.get_epoch(),
            # from validation journal
            'process_start': datetime.now(),
            'process_end': datetime.now(),
            'validation_loss': 0.0,
        }

        if self.log_pattern:
            try:
                self.log_pattern.format(**template_before_process)
            except Exception as e:
                raise e
        if self.journal_pattern:
            try:
                self.journal_pattern.format(**template_after_process)
            except Exception as e:
                raise e

    def validate(self):
        _validate(self)


class ValidationJournal(object):
    def __init__(self,
                 process_start : datetime,
                 process_end : datetime,
                 validation_loss : float,
                 validation_step_journals : list,
                 log_path : str,
                 spec : ValidationSpec):

        self.process_start = process_start
        self.process_end = process_end
        self.validation_loss = validation_loss
        self.validation_step_journals = validation_step_journals
        self.log_path = log_path
        self.spec = spec

    def to_dict(self):
        return {
            'process_start': self.process_start.isoformat(),
            'process_end': self.process_end.isoformat(),
            'validation_loss': self.validation_loss,
            'validation_step_journals': [
                e.to_dict() for e in self.validation_step_journals
            ],
            'log_path': self.log_path,
            'spec': self.spec.to_dict()
        }

    @classmethod
    def from_dict(cls, d : dict):
        return cls(
            process_start=datetime.fromisoformat(d['process_start']),
            process_end=datetime.fromisoformat(d['process_end']),
            validation_loss=d['validation_loss'],
            validation_step_journals=[
                utils.StepJournal.from_dict(e)
                for e in d.get('validation_step_journals')
            ],
            log_path=d['log_path'],
            spec=ValidationSpec.from_dict(d['spec']),
        )


