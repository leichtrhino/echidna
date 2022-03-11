
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


