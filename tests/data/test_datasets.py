
import unittest
import os
import json
import pathlib
import tempfile

from echidna.data.augmentations import AugmentationSpec
from echidna.data.mixtures import MixSetSpec
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

        aug_spec = AugmentationSpec(
            source_sample_rate=8000,
            target_sample_rate=8000,
            waveform_length=int(8000 * 0.5),

            scale_range=[0.1, 1.1],
            scale_point_range=[1, 5],
            time_stretch_range=[0.5, 1.5],
            pitch_shift_range=[0.5, 1.5],
            seed=self.seed,
            augmentation_per_parent=2,
            input_metadata_path=self.sample_dir/'e1'/'metadata.json',
            output_metadata_path=self.sample_dir/'e1'/'metadata_aug.json',
            journal_path=self.sample_dir/'e1'/'journal_aug.json',
            log_path=self.sample_dir/'e1'/'log_aug.txt',
            log_level='INFO',
            jobs=None
        )
        aug_spec.save_augmentation()

        mix_spec = MixSetSpec(
            mix_category_list=[
                {
                    'category': ['ct001'],
                    'min_channel': 1,
                    'max_channel': 3,
                },
                {
                    'category': ['ct002', 'other'],
                    'min_channel': 1,
                    'max_channel': 3,
                },
            ],
            seed=self.seed,
            mix_per_parent=2,
            input_metadata_path=self.sample_dir/'e1'/'metadata_aug.json',
            output_metadata_path=self.sample_dir/'e1'/'metadata_mix.json',
            journal_path=self.sample_dir/'e1'/'journal_mix.json',
            log_path=self.sample_dir/'e1'/'log_mix.txt',
            log_level='INFO',
            jobs=None
        )
        mix_spec.save_mixture()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_dataset(self):
        # initialize dataset
        dataset = Dataset([self.sample_dir/'e1'/'metadata_mix.json'])
        self.assertTrue(len(dataset), 2 * 2 * 3)
        for data_list, metadata_list in dataset:
            for data in data_list:
                wave = data['wave']
                self.assertTrue(wave.shape, (int(8000 * 0.5),))

        # serialize/deserialize
        dataset_dict = dataset.to_dict()
        dataset_from_dict = Dataset.from_dict(dataset_dict)
        self.assertEqual(dataset.to_dict(), dataset_from_dict.to_dict())

