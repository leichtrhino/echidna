
import typing as tp
import inspect
import os
from datetime import datetime
import logging
import json
import random
import torch

from ..data.datasets import Dataset
from ..data.dataloaders import build_dataloader
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
            'process_finish': datetime.now(),
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

class EpochJournal(object):
    def __init__(self,
                 process_start : datetime,
                 process_finish : datetime,
                 model_epoch : int,
                 training_epoch : int,
                 training_loss : float,
                 training_step_journals : list,
                 validation_loss : float,
                 validation_step_journals : list,
                 checkpoint_path : str,
                 log_path : str,
                 spec : TrainingSpec):

        self.process_start = process_start
        self.process_finish = process_finish
        self.model_epoch = model_epoch
        self.training_epoch = training_epoch
        self.training_loss = training_loss
        self.training_step_journals = training_step_journals
        self.validation_loss = validation_loss
        self.validation_step_journals = validation_step_journals
        self.checkpoint_path = checkpoint_path
        self.log_path = log_path
        self.spec = spec

    def to_dict(self):
        return {
            'process_start': self.process_start.isoformat(),
            'process_finish': self.process_finish.isoformat(),
            'model_epoch': self.model_epoch,
            'training_epoch': self.training_epoch,
            'training_loss': self.training_loss,
            'training_step_journals': [
                e.to_dict() for e in self.training_step_journals
            ],
            'validation_loss': self.validation_loss,
            'validation_step_journals': [
                e.to_dict() for e in self.validation_step_journals
            ] if self.validation_step_journals else None,
            'checkpoint_path': self.checkpoint_path,
            'log_path': self.log_path,
            'spec': self.spec.to_dict()
        }

    @classmethod
    def from_dict(cls, d : dict):
        return cls(
            process_start=datetime.fromisoformat(d['process_start']),
            process_finish=datetime.fromisoformat(d['process_finish']),
            model_epoch=d['model_epoch'],
            training_epoch=d['training_epoch'],
            training_loss=d['training_loss'],
            training_step_journals=[
                utils.StepJournal.from_dict(e)
                for e in d['training_step_journals']
            ],
            validation_loss=d.get('validation_loss'),
            validation_step_journals=[
                utils.StepJournal.from_dict(e)
                for e in d.get('validation_step_journals')
            ] if d.get('validation_step_journals') is not None else None,
            checkpoint_path=d['checkpoint_path'],
            log_path=d['log_path'],
            spec=TrainingSpec.from_dict(d['spec']),
        )

def _train(spec : TrainingSpec):
    for training_epoch in range(1, spec.training_epochs+1):
        _train_epoch(spec, training_epoch)

def _train_epoch(spec : TrainingSpec, training_epoch):
    """
    """

    process_start = datetime.now()

    # get log path and initialize log handler
    logger = None
    if spec.log_pattern:
        model_epoch_at_end = spec.checkpoint.get_epoch() + 1
        pattern_dict ={
            # from specification
            'model': spec.checkpoint.get_model_class(),
            'model_epoch': model_epoch_at_end,
            'optimizer': spec.checkpoint.get_optimizer_class(),
            'lr': spec.checkpoint.get_torch_optimizer().defaults['lr'],
            'scheduler': spec.checkpoint.get_scheduler_class(),
            # from epoch journal
            'process_start': datetime.now(),
            'training_epoch': training_epoch,
        }

        logger = logging.getLogger(__name__)
        log_path = spec.log_pattern.format(**pattern_dict)
        logger.setLevel(spec.log_level)
        if not os.path.isdir(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(
            logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)

    # generate seed for dataloader
    random.seed(spec.seed)
    dataloader_seed = [
        random.randrange(2**32)
        for _ in range(spec.checkpoint.get_epoch()+1)
    ][-1]

    if logger:
        logger.info(json.dumps({
            'type': 'start_epoch',
            'timestamp': datetime.now().isoformat(),
            'model_class': spec.checkpoint.get_model_class(),
            'model_epoch': spec.checkpoint.get_epoch(),
            'training_epoch': training_epoch,
        }))

    # training loop
    if logger:
        logger.info(json.dumps({
            'type': 'start_training_steps',
            'timestamp': datetime.now().isoformat(),
            'model_class': spec.checkpoint.get_model_class(),
            'model_epoch': spec.checkpoint.get_epoch(),
            'training_epoch': training_epoch,
        }))


    training_loader = build_dataloader(
        spec.training_dataset,
        spec.training_sample_size,
        spec.batch_size,
        spec.jobs,
        shuffle=True,
        seed=dataloader_seed)
    training_step_journals = []

    # prepare model and optimizer
    spec.checkpoint.get_torch_model().train()
    spec.checkpoint.get_torch_model().to(spec.device)
    for state in spec.checkpoint.get_torch_optimizer().state.values():
        for k, v in state.items():
            if type(v) == torch.Tensor:
                state[k] = v.to(spec.device)

    torch.manual_seed(dataloader_seed)
    for step, (data, metadata) in enumerate(training_loader, 1):
        step_journal = utils.process_batch(
            spec, training_epoch, step, data, metadata, logger)
        step_journal.sample_indices = metadata['index']
        training_step_journals.append(step_journal)

    training_loss = sum(j.batch_loss for j in training_step_journals) \
        / sum(len(j.sample_losses) for j in training_step_journals)

    if logger:
        logger.info(json.dumps({
            'type': 'end_training_steps',
            'timestamp': datetime.now().isoformat(),
            'model_class': spec.checkpoint.get_model_class(),
            'model_epoch': spec.checkpoint.get_epoch(),
            'training_epoch': training_epoch,
            'training_loss': training_loss,
        }))

    # validation loop (if available)
    validation_step_journals = None
    validation_loss = None
    if spec.validation_dataset:
        if logger:
            logger.info(json.dumps({
                'type': 'start_validation_steps',
                'timestamp': datetime.now().isoformat(),
                'model_class': spec.checkpoint.get_model_class(),
                'model_epoch': spec.checkpoint.get_epoch(),
                'training_epoch': training_epoch,
            }))

        validation_loader = build_dataloader(
            spec.validation_dataset,
            spec.validation_sample_size,
            spec.batch_size,
            spec.jobs,
            shuffle=False,
            seed=dataloader_seed)
        validation_step_journals = []

        spec.checkpoint.get_torch_model().eval()
        for step, (data, metadata) in enumerate(validation_loader, 1):
            with torch.no_grad():
                step_journal = utils.process_batch(
                    spec, training_epoch, step, data, metadata, logger)
            step_journal.sample_indices = metadata['index']
            validation_step_journals.append(step_journal)

        validation_loss = sum(j.batch_loss for j in validation_step_journals)\
            / sum(len(j.sample_losses) for j in validation_step_journals)

        if logger:
            logger.info(json.dumps({
                'type': 'end_validation_steps',
                'timestamp': datetime.now().isoformat(),
                'model_class': spec.checkpoint.get_model_class(),
                'model_epoch': spec.checkpoint.get_epoch(),
                'training_epoch': training_epoch,
                'validation_loss': validation_loss,
            }))

    # move checkpoint back to cpu
    spec.checkpoint.get_torch_model().to('cpu')
    for state in spec.checkpoint.get_torch_optimizer().state.values():
        for k, v in state.items():
            if type(v) == torch.Tensor:
                state[k] = v.to('cpu')

    # step scheduler
    scheduler = spec.checkpoint.get_torch_scheduler()
    if 'metrics' in inspect.signature(scheduler.step).parameters:
        # for ReduceLROnPlateau
        metric = validation_loss if validation_loss is not None \
            else training_loss
        scheduler.step(metrics=metric)
    else:
        scheduler.step()

    logger.info(json.dumps({
        'type': 'step_scheduler',
        'timestamp': datetime.now().isoformat(),
        'model_class': spec.checkpoint.get_model_class(),
        'model_epoch': spec.checkpoint.get_epoch(),
        'training_epoch': training_epoch,
    }))

    # finish epoch
    process_finish = datetime.now()

    # get dict for formatting checkpoint and journal file
    pattern_dict = {
        # from specification
        'model': spec.checkpoint.get_model_class(),
        'model_epoch': spec.checkpoint.get_epoch(),
        'optimizer': spec.checkpoint.get_optimizer_class(),
        'lr': spec.checkpoint.get_torch_optimizer().defaults['lr'],
        'scheduler': spec.checkpoint.get_scheduler_class(),
        # from epoch journal
        'process_start': process_start,
        'process_finish': process_finish,
        'training_epoch': training_epoch,
        'training_loss': training_loss,
        'validation_loss': validation_loss,
    }

    # save checkpoint
    checkpoint_path = spec.checkpoint_pattern.format(**pattern_dict)
    if not os.path.isdir(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    spec.checkpoint.save_torch_checkpoint(checkpoint_path)

    logger.info(json.dumps({
        'type': 'save_checkpoint',
        'timestamp': datetime.now().isoformat(),
        'model_class': spec.checkpoint.get_model_class(),
        'model_epoch': spec.checkpoint.get_epoch(),
        'training_epoch': training_epoch,
        'checkpoint_path': checkpoint_path,
    }))

    # save journal
    if spec.journal_pattern:
        journal_path = spec.journal_pattern.format(**pattern_dict)
        journal = EpochJournal(
            process_start=process_start,
            process_finish=process_finish,
            model_epoch=spec.checkpoint.get_epoch(),
            training_epoch=training_epoch,
            training_loss=training_loss,
            training_step_journals=training_step_journals,
            validation_loss=validation_loss,
            validation_step_journals=validation_step_journals,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            spec=spec,
        )
        if not os.path.isdir(os.path.dirname(journal_path)):
            os.makedirs(os.path.dirname(journal_path), exist_ok=True)
        with open(journal_path, 'w') as fp:
            json.dump(journal.to_dict(), fp)

        logger.info(json.dumps({
            'type': 'save_journal',
            'timestamp': datetime.now().isoformat(),
            'model_class': spec.checkpoint.get_model_class(),
            'model_epoch': spec.checkpoint.get_epoch(),
            'training_epoch': training_epoch,
            'journal_path': journal_path,
        }))

    # finish epoch and close log handler
    if logger:
        logger.info(json.dumps({
            'type': 'end_epoch',
            'timestamp': datetime.now().isoformat(),
            'model_class': spec.checkpoint.get_model_class(),
            'model_epoch': spec.checkpoint.get_epoch(),
            'training_epoch': training_epoch,
        }))
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()

