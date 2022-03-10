
import unittest
import logging
import tempfile
import pathlib
import json
import math
import random
import torch

from echidna.procs.trainings import TrainingSpec
from echidna.procs.validations import ValidationSpec
from echidna.procs.utils import (
    build_dataloader,
    StepJournal,
    process_batch
)
from echidna.metrics.waveform import L1WaveformLoss, L2WaveformLoss
from echidna.metrics.deepclustering import DeepClusteringLoss
from echidna.metrics.composite import CompositeLoss

from .utils import ToyDataset, get_training_spec, get_validation_spec
from ..models.utils import get_initial_model, get_initial_checkpoint

class TestDataLoader(unittest.TestCase):
    def test_init_ok(self):
        dataset = ToyDataset(6, seed=140343)
        # shuffle = False
        loader = build_dataloader(dataset,
                                  batch_size=4,
                                  sample_size=None,
                                  num_workers=0,
                                  shuffle=False,
                                  seed=None)
        for indices, (data, metadata) in zip([[0, 1, 2, 3], [4, 5]], loader):
            self.assertEqual(data['waves'].shape, (len(indices), 3, 4000))
            for i, w in zip(indices, data['waves']):
                self.assertEqual(i, w[0][0].item())

        # shuffle = True
        loader = build_dataloader(dataset,
                                  sample_size=None,
                                  batch_size=4,
                                  num_workers=0,
                                  shuffle=True,
                                  seed=1410343)
        metadata_list_1 = [metadata for data, metadata in loader]
        wave_1 = [data['waves'] for data, metadata in loader]
        loader = build_dataloader(dataset,
                                  sample_size=None,
                                  batch_size=4,
                                  num_workers=0,
                                  shuffle=True,
                                  seed=1410343)
        metadata_list_2 = [metadata for data, metadata in loader]
        wave_2 = [data['waves'] for data, metadata in loader]
        self.assertEqual(metadata_list_1, metadata_list_2)
        self.assertEqual(len(wave_1), len(wave_2))
        for w1, w2 in zip(wave_1, wave_2):
            self.assertTrue(torch.all(w1 == w2))

        # partial dataset
        loader = build_dataloader(dataset,
                                  batch_size=4,
                                  sample_size=5,
                                  num_workers=0,
                                  shuffle=False,
                                  seed=None)
        for indices, (data, metadata) in zip([[0, 1, 2, 3], [4]], loader):
            self.assertEqual(data['waves'].shape, (len(indices), 3, 4000))
            for i, w in zip(indices, data['waves']):
                self.assertEqual(i, w[0][0].item())

class TestStep(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _test_process_batch_training(self,
                                     name,
                                     chimera,
                                     composite_loss,
                                     dataset_size,
                                     sample_size,
                                     batch_size,
                                     compute_batch_size):

        tmproot = pathlib.Path(self.tmpdir.name) / name
        tmproot.mkdir()

        spec = get_training_spec(
            chimera=chimera,
            composite_loss=composite_loss,
            dataset_size=dataset_size,
            sample_size=sample_size,
            batch_size=batch_size,
            compute_batch_size=compute_batch_size,
            max_grad_norm=1e4,
            checkpoint_pattern=tmproot \
            / '{model}-{model_epoch}-{process_end}.tar',
            journal_pattern=tmproot \
            / '{model}-{model_epoch}-{process_end}.json',
            log_pattern=tmproot \
            / '{model}-{model_epoch}.log'
        )

        data, metadata = next(iter(build_dataloader(
            spec.training_dataset,
            sample_size=sample_size,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            seed=1410343
        )))
        self.assertEqual(type(data), dict)
        self.assertEqual(type(metadata), dict)

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(tmproot / 'step_log.log')
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)

        step_journal = process_batch(spec,
                                     epoch=1,
                                     step=11,
                                     data=data,
                                     metadata=metadata,
                                     logger=logger)

        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()

        # test journal serialization
        step_journal_ = StepJournal.from_dict(step_journal.to_dict())
        self.assertEqual(step_journal.to_dict(), step_journal_.to_dict())

        # journal has specified epoch, step
        self.assertEqual(step_journal.step, 11)
        # journal has specified sample_losses
        self.assertEqual(len(step_journal.sample_losses), batch_size)
        for l in step_journal.sample_losses:
            if composite_loss:
                self.assertEqual(
                    l.keys(),
                    {'l1_waveform', 'l2_waveform', 'deep_clustering'}
                )
            else:
                self.assertEqual(l.keys(), {'l1_waveform'})
        # journal has specified sample_indices
        self.assertEqual(
            step_journal.sample_indices, list(range(batch_size)))

        true_event_types = ['start_step'] \
            + ['get_samples', 'compute_inference',
               'align_samples', 'compute_loss'] \
               * math.ceil(batch_size / compute_batch_size) \
            + ['update_model', 'update_model_detail', 'end_step']

        with open(tmproot / 'step_log.log', 'r') as fp:
            lc = 0
            tot_sample = 0
            for line in fp:
                idx = line.index(' ')
                event_dict = json.loads(line[idx+1:])
                self.assertEqual(event_dict['type'], true_event_types[lc])
                self.assertEqual(event_dict['mode'], 'training')
                self.assertEqual(event_dict['training_epoch'], 1)
                self.assertEqual(event_dict['model_class'],
                                 spec.checkpoint.get_model_class())
                self.assertEqual(event_dict['model_epoch'], 0)
                self.assertEqual(event_dict['step'], 11)
                if event_dict['type'] == 'compute_loss':
                    tot_sample += len(event_dict['losses'])
                lc += 1
            self.assertEqual(lc, len(true_event_types))
            self.assertEqual(tot_sample, batch_size)



        # test for validation
        spec.checkpoint.get_torch_model().eval()

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(tmproot / 'step_log.log', mode='w')
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)

        step_journal = process_batch(spec,
                                     epoch=1,
                                     step=11,
                                     data=data,
                                     metadata=metadata,
                                     logger=logger)

        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()

        true_event_types = ['start_step'] \
            + ['get_samples', 'compute_inference',
               'align_samples', 'compute_loss'] \
               * math.ceil(batch_size / compute_batch_size) \
            + ['end_step']

        with open(tmproot / 'step_log.log', 'r') as fp:
            lc = 0
            tot_sample = 0
            for line in fp:
                idx = line.index(' ')
                event_dict = json.loads(line[idx+1:])
                self.assertEqual(event_dict['type'], true_event_types[lc])
                self.assertEqual(event_dict['mode'], 'validation')
                self.assertEqual(event_dict['training_epoch'], 1)
                self.assertEqual(event_dict['model_class'],
                                 spec.checkpoint.get_model_class())
                self.assertEqual(event_dict['model_epoch'], 0)
                self.assertEqual(event_dict['step'], 11)
                if event_dict['type'] == 'compute_loss':
                    tot_sample += len(event_dict['losses'])
                lc += 1
            self.assertEqual(lc, len(true_event_types))
            self.assertEqual(tot_sample, batch_size)


    def _test_process_batch_validation(self,
                                       name,
                                       chimera,
                                       composite_loss,
                                       dataset_size,
                                       sample_size,
                                       batch_size,
                                       compute_batch_size):

        tmproot = pathlib.Path(self.tmpdir.name) / name
        tmproot.mkdir()

        spec = get_validation_spec(
            chimera=chimera,
            composite_loss=composite_loss,
            dataset_size=dataset_size,
            sample_size=sample_size,
            batch_size=batch_size,
            compute_batch_size=compute_batch_size,
            journal_pattern=tmproot \
            / '{model}-{model_epoch}-{process_end}.json',
            log_pattern=tmproot \
            / '{model}-{model_epoch}.log'
        )

        data, metadata = next(iter(build_dataloader(
            spec.validation_dataset,
            sample_size=sample_size,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            seed=1410343
        )))
        self.assertEqual(type(data), dict)
        self.assertEqual(type(metadata), dict)

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(tmproot / 'step_log.log')
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)

        step_journal = process_batch(spec,
                                     epoch=1,
                                     step=11,
                                     data=data,
                                     metadata=metadata,
                                     logger=logger)

        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()

        # test journal serialization
        step_journal_ = StepJournal.from_dict(step_journal.to_dict())
        self.assertEqual(step_journal.to_dict(), step_journal_.to_dict())

        # journal has specified epoch, step
        self.assertEqual(step_journal.step, 11)
        # journal has specified sample_losses
        self.assertEqual(len(step_journal.sample_losses), batch_size)
        for l in step_journal.sample_losses:
            if composite_loss:
                self.assertEqual(
                    l.keys(),
                    {'l1_waveform', 'l2_waveform', 'deep_clustering'}
                )
            else:
                self.assertEqual(l.keys(), {'l1_waveform'})
        # journal has specified sample_indices
        self.assertEqual(
            step_journal.sample_indices, list(range(batch_size)))

        true_event_types = ['start_step'] \
            + ['get_samples', 'compute_inference',
               'align_samples', 'compute_loss'] \
               * math.ceil(batch_size / compute_batch_size) \
            + ['end_step']

        with open(tmproot / 'step_log.log', 'r') as fp:
            lc = 0
            tot_sample = 0
            for line in fp:
                idx = line.index(' ')
                event_dict = json.loads(line[idx+1:])
                self.assertEqual(event_dict['type'], true_event_types[lc])
                self.assertEqual(event_dict['mode'], 'validation')
                self.assertEqual(event_dict['model_class'],
                                 spec.model.get_class())
                self.assertEqual(event_dict['model_epoch'], 0)
                self.assertEqual(event_dict['step'], 11)
                if event_dict['type'] == 'compute_loss':
                    tot_sample += len(event_dict['losses'])
                lc += 1
            self.assertEqual(lc, len(true_event_types))
            self.assertEqual(tot_sample, batch_size)


    def test_process_batch_training_ok(self):
        # basic setup
        self._test_process_batch_training(name='encdec',
                                          chimera=False,
                                          composite_loss=False,
                                          dataset_size=12,
                                          sample_size=12,
                                          batch_size=4,
                                          compute_batch_size=4)

        self._test_process_batch_training(name='chimera-wave',
                                          chimera=True,
                                          composite_loss=False,
                                          dataset_size=12,
                                          sample_size=12,
                                          batch_size=4,
                                          compute_batch_size=4)

        # composite loss
        self._test_process_batch_training(name='chimera',
                                          chimera=True,
                                          composite_loss=True,
                                          dataset_size=12,
                                          sample_size=12,
                                          batch_size=4,
                                          compute_batch_size=4)

        # partial data
        self._test_process_batch_training(name='encdec-oddbatch',
                                          chimera=False,
                                          composite_loss=False,
                                          dataset_size=12,
                                          sample_size=12,
                                          batch_size=5,
                                          compute_batch_size=4)

    def test_process_batch_training_ng(self):
        # no embd output despite of the loss  requires embd
        is_error = False
        try:
            self._test_process_batch_training(name='encdec-no-embd',
                                              chimera=False,
                                              composite_loss=True,
                                              dataset_size=12,
                                              sample_size=12,
                                              batch_size=4,
                                              compute_batch_size=4)
        except Exception:
            is_error = True
        self.assertTrue(is_error)

    def test_process_batch_validation_ok(self):
        # basic setup
        self._test_process_batch_validation(name='encdec-val',
                                            chimera=False,
                                            composite_loss=False,
                                            dataset_size=12,
                                            sample_size=12,
                                            batch_size=4,
                                            compute_batch_size=4)

        self._test_process_batch_validation(name='chimera-wave-val',
                                            chimera=True,
                                            composite_loss=False,
                                            dataset_size=12,
                                            sample_size=12,
                                            batch_size=4,
                                            compute_batch_size=4)

        # composite loss
        self._test_process_batch_validation(name='chimera-val',
                                            chimera=True,
                                            composite_loss=True,
                                            dataset_size=12,
                                            sample_size=12,
                                            batch_size=4,
                                            compute_batch_size=4)

        # partial data
        self._test_process_batch_validation(name='encdec-oddbatch-val',
                                            chimera=False,
                                            composite_loss=False,
                                            dataset_size=12,
                                            sample_size=12,
                                            batch_size=5,
                                            compute_batch_size=4)

    def test_process_batch_validation_ng(self):
        # no embd output despite of the loss  requires embd
        is_error = False
        try:
            self._test_process_batch_validation(name='encdec-no-embd-val',
                                                chimera=False,
                                                composite_loss=True,
                                                dataset_size=12,
                                                sample_size=12,
                                                batch_size=4,
                                                compute_batch_size=4)
        except Exception:
            is_error = True
        self.assertTrue(is_error)

