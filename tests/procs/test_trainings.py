
import unittest
import logging
import tempfile
import pathlib
import json
import math
import torch

from echidna.models.checkpoints import SavedCheckpoint
from echidna.procs.trainings import TrainingSpec, EpochJournal

from .utils import get_training_spec

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _test_training(self,
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
            / '{model}-{model_epoch}-{process_finish}.tar',
            journal_pattern=tmproot \
            / '{model}-{model_epoch}-{process_finish}.json',
            log_pattern=tmproot \
            / '{model}-{model_epoch}.log'
        )

        spec.train()

        for e in range(1, spec.training_epochs):
            prefix = f'{spec.checkpoint.get_model_class()}-{e}'
            # check checkpoint
            checkpoint_path = next(tmproot.glob(prefix+'*.tar'))
            checkpoint = SavedCheckpoint(checkpoint_path)
            self.assertEqual(checkpoint.get_epoch(), e)

            # check journal
            journal_path = next(tmproot.glob(prefix+'*.json'))
            with open(journal_path, 'r') as fp:
                journal = json.load(fp)
            self.assertEqual(journal['model_epoch'], e)
            self.assertEqual(len(journal['training_step_journals']),
                             math.ceil(sample_size / batch_size))
            self.assertEqual(sum(len(j['sample_losses'])
                                 for j in journal['training_step_journals']),
                             sample_size)
            self.assertEqual(len(journal['validation_step_journals']),
                             math.ceil(sample_size / batch_size))
            self.assertEqual(
                sum(len(j['sample_losses'])
                    for j in journal['validation_step_journals']),
                sample_size)

            # check log
            true_event_types = []
            true_event_types.extend(['start_epoch', 'start_training_steps'])
            remaining_sample_size = sample_size
            while remaining_sample_size > 0:
                true_event_types.append('start_step')
                remaining_batch_size = min(remaining_sample_size, batch_size)
                while remaining_batch_size > 0:
                    true_event_types.extend([
                        'get_samples', 'compute_inference',
                        'align_samples', 'compute_loss'
                    ])
                    remaining_batch_size -= compute_batch_size
                true_event_types.extend([
                    'update_model', 'update_model_detail', 'end_step'
                ])
                remaining_sample_size -= batch_size
            true_event_types.extend(
                ['end_training_steps', 'start_validation_steps'])
            remaining_sample_size = sample_size
            while remaining_sample_size > 0:
                true_event_types.append('start_step')
                remaining_batch_size = min(remaining_sample_size, batch_size)
                while remaining_batch_size > 0:
                    true_event_types.extend([
                        'get_samples', 'compute_inference',
                        'align_samples', 'compute_loss'
                    ])
                    remaining_batch_size -= compute_batch_size
                true_event_types.extend(['end_step'])
                remaining_sample_size -= batch_size
            true_event_types.extend(['end_validation_steps', 'step_scheduler',
                                     'save_checkpoint', 'save_journal',
                                     'end_epoch'])

            log_path = next(tmproot.glob(prefix+'*.log'))
            with open(log_path, 'r') as fp:
                lc = 0
                total_samples = 0
                for l in fp:
                    event_dict = json.loads(l[l.index(' ')+1:])
                    self.assertEqual(event_dict['type'], true_event_types[lc])
                    if event_dict['type'] == 'compute_loss':
                        total_samples += len(event_dict['losses'])
                    lc += 1
            self.assertEqual(lc, len(true_event_types))
            # NOTE: double the sample size for training and validations
            self.assertEqual(total_samples, sample_size + sample_size)


    def test_training_ok(self):
        # basic setup
        self._test_training(name='encdec',
                            chimera=False,
                            composite_loss=False,
                            dataset_size=12,
                            sample_size=12,
                            batch_size=4,
                            compute_batch_size=4)

        # partial data
        self._test_training(name='encdec-oddbatch',
                            chimera=False,
                            composite_loss=False,
                            dataset_size=12,
                            sample_size=10,
                            batch_size=4,
                            compute_batch_size=4)

    def test_training_ng(self):
        # no embd output despite of the loss  requires embd
        is_error = False
        try:
            self._test_training(name='encdec-no-embd',
                                chimera=False,
                                composite_loss=True,
                                dataset_size=12,
                                sample_size=12,
                                batch_size=4,
                                compute_batch_size=4)
        except Exception:
            is_error = True
        self.assertTrue(is_error)

