
import unittest
import os
import pathlib
import tempfile

from echidna.data.samples import Sample
from echidna.data.augmentations import Augmentation, AugmentationSpec
from echidna.data.mixtures import Mixture, MixtureSpec
from echidna.data.datasets import Dataset

from .utils import prepare_datasources

class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.seed = 140343
        self.samplespecs = prepare_datasources(self.tmpdir, self.seed)

        self.sample_dir = pathlib.Path(self.tmpdir.name) / 'samples'
        self.samplespec = self.samplespecs['E1']
        self.samplespec.save_samples()

        self.aug_dir = pathlib.Path(self.tmpdir.name) / 'aug'
        aug_spec = AugmentationSpec(
            algorithm_name='random',
            algorithm_params={
                'source_sample_rate': 8000,
                'target_sample_rate': 8000,
                'waveform_length': int(8000 * 0.5),

                'normalize': True,
                'scale_range': [0.1, 1.1],
                'scale_point_range': [1, 5],
                'time_stretch_range': [0.5, 1.5],
                'pitch_shift_range': [0.5, 1.5],
            },
            seed=self.seed,
            augmentation_per_sample=2,
            sample_metadata_path=self.sample_dir/'e1'/'metadata.json',
            augmentation_metadata_path=self.aug_dir/'e1'/'metadata.json',
            journal_path=self.aug_dir/'e1'/'journal.json',
            jobs=None
        )
        aug_spec.save_augmentation()

        self.mix_dir = pathlib.Path(self.tmpdir.name) / 'mix'
        mix_spec = MixtureSpec(
            algorithm_name='category',
            algorithm_params={
                'mix_category_list': [['ct001'], ['ct002', 'ct003']],
                'include_other': False,
            },
            seed=self.seed,
            mix_per_sample=2,
            sample_metadata_path=self.sample_dir/'e1'/'metadata.json',
            mixture_metadata_path=self.mix_dir/'e1'/'metadata.json',
            journal_path=self.mix_dir/'e1'/'journal.json',
            jobs=None
        )
        mix_spec.save_mixture()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_dataset(self):
        # initialize dataset
        dataset = Dataset(
            samples_metadata_path=self.sample_dir/'e1'/'metadata.json',
            augmentations_metadata_path=self.aug_dir/'e1'/'metadata.json',
            mixtures_metadata_path=self.mix_dir/'e1'/'metadata.json',
        )

        self.assertTrue(len(dataset), 2 * 2 * 3)
        for data, metadata in dataset:
            waves = data['waves']
            self.assertTrue(waves.shape, (2, int(8000 * 0.5)))


