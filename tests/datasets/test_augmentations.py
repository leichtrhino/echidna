
import unittest
import pathlib
import tempfile
import json

from echidna.datasets.augmentations import (
    Augmentation,
    AugmentationsJournal,
    AugmentationSpec
)
from echidna.datasets.samples import (
    Sample,
    SampleSpec
)

from .utils import prepare_datasources

class TestAugmentations(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.seed = 140343
        self.samplespecs = prepare_datasources(self.tmpdir, self.seed)
        self.samplespec = self.samplespecs['E1']
        self.samplespec.save_samples()
        self.sample_dir = pathlib.Path(self.tmpdir.name) / 'samples'

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_random_augmentation(self):
        aug_dir = pathlib.Path(self.tmpdir.name) / 'augmentation_1'
        spec = AugmentationSpec(
            algorithm_name='RandomAugmentation',
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
            augmentation_metadata_path=aug_dir/'e1'/'metadata.json',
            journal_path=aug_dir/'e1'/'journal.json',
            jobs=None
        )
        spec.save_augmentation()

        # load metadata
        with open(spec.augmentation_metadata_path, 'r') as fp:
            augmentations = Augmentation.from_list(json.load(fp))

        self.assertEqual(len(augmentations), 2)
        for a in augmentations:
            self.assertEqual(a.sample_index, 0)
            self.assertEqual(a.source_sample_rate, 8000)
            self.assertEqual(a.target_sample_rate, 8000)
            self.assertEqual(a.waveform_length, int(8000 * 0.5))
            self.assertEqual(len(a.offsets), 3)
            self.assertEqual(a.offsets[0], a.offsets[1]) # sync check
            self.assertEqual(len(a.time_stretch_rates), 3)
            self.assertEqual(a.time_stretch_rates[0],
                             a.time_stretch_rates[1]) # sync check
            self.assertEqual(len(a.pitch_shift_rates), 3)
            self.assertEqual(len(a.scale_amount_list), 3)
            self.assertEqual(len(a.scale_fraction_list), 3)

        # load journal
        with open(spec.journal_path, 'r') as fp:
            journal = AugmentationsJournal.from_dict(json.load(fp))

        self.assertEqual(journal.metadata_path, 'metadata.json')
        self.assertEqual(journal.spec.to_dict(), spec.to_dict())

    def test_entropy_augmentation_minscore(self):
        aug_dir = pathlib.Path(self.tmpdir.name) / 'augmentation_2'
        spec = AugmentationSpec(
            algorithm_name='EntropyAugmentation',
            algorithm_params={
                'source_sample_rate': 8000,
                'target_sample_rate': 8000,
                'waveform_length': int(8000 * 0.5),

                'normalize': True,
                'scale_range': [0.1, 1.1],
                'scale_point_range': [1, 5],
                'time_stretch_range': [0.5, 1.5],
                'pitch_shift_range': [0.5, 1.5],

                'mixture_algorithm_name': 'CategoryMix',
                'mixture_algorithm_params': {
                    'mix_category_list': [
                        ['ct001'],
                        ['ct002', 'ct003']
                    ],
                    'include_other': False,
                },
                'trials_per_augmentation': 10,
                'separation_difficulty': 0.0,

            },
            seed=self.seed,
            augmentation_per_sample=2,
            sample_metadata_path=self.sample_dir/'e1'/'metadata.json',
            augmentation_metadata_path=aug_dir/'e1'/'metadata.json',
            journal_path=aug_dir/'e1'/'journal.json',
            jobs=None
        )
        spec.save_augmentation()

        # load metadata
        with open(spec.augmentation_metadata_path, 'r') as fp:
            augmentations = Augmentation.from_list(json.load(fp))

        self.assertEqual(len(augmentations), 2)
        for a in augmentations:
            self.assertEqual(a.sample_index, 0)
            self.assertEqual(a.source_sample_rate, 8000)
            self.assertEqual(a.target_sample_rate, 8000)
            self.assertEqual(a.waveform_length, int(8000 * 0.5))
            self.assertEqual(len(a.offsets), 3)
            self.assertEqual(a.offsets[0], a.offsets[1]) # sync check
            self.assertEqual(len(a.time_stretch_rates), 3)
            self.assertEqual(a.time_stretch_rates[0],
                             a.time_stretch_rates[1]) # sync check
            self.assertEqual(len(a.pitch_shift_rates), 3)
            self.assertEqual(len(a.scale_amount_list), 3)
            self.assertEqual(len(a.scale_fraction_list), 3)

        # load journal
        with open(spec.journal_path, 'r') as fp:
            journal = AugmentationsJournal.from_dict(json.load(fp))

        self.assertEqual(journal.metadata_path, 'metadata.json')
        self.assertEqual(journal.spec.to_dict(), spec.to_dict())
        for j in journal.augmentation_journals:
            self.assertEqual(j.algorithm_out['score'],
                             j.algorithm_out['scoreStats']['min'])


    def test_entropy_augmentation_maxscore(self):
        aug_dir = pathlib.Path(self.tmpdir.name) / 'augmentation_3'
        spec = AugmentationSpec(
            algorithm_name='EntropyAugmentation',
            algorithm_params={
                'source_sample_rate': 8000,
                'target_sample_rate': 8000,
                'waveform_length': int(8000 * 0.5),

                'normalize': True,
                'scale_range': [0.1, 1.1],
                'scale_point_range': [1, 5],
                'time_stretch_range': [0.5, 1.5],
                'pitch_shift_range': [0.5, 1.5],

                'mixture_algorithm_name': 'CategoryMix',
                'mixture_algorithm_params': {
                    'mix_category_list': [
                        ['ct001'],
                        ['ct002', 'ct003']
                    ],
                    'include_other': False,
                },
                'trials_per_augmentation': 10,
                'separation_difficulty': 1.0,

            },
            seed=self.seed,
            augmentation_per_sample=2,
            sample_metadata_path=self.sample_dir/'e1'/'metadata.json',
            augmentation_metadata_path=aug_dir/'e1'/'metadata.json',
            journal_path=aug_dir/'e1'/'journal.json',
            jobs=None
        )
        spec.save_augmentation()

        # load metadata
        with open(spec.augmentation_metadata_path, 'r') as fp:
            augmentations = Augmentation.from_list(json.load(fp))

        self.assertEqual(len(augmentations), 2)
        for a in augmentations:
            self.assertEqual(a.sample_index, 0)
            self.assertEqual(a.source_sample_rate, 8000)
            self.assertEqual(a.target_sample_rate, 8000)
            self.assertEqual(a.waveform_length, int(8000 * 0.5))
            self.assertEqual(len(a.offsets), 3)
            self.assertEqual(a.offsets[0], a.offsets[1]) # sync check
            self.assertEqual(len(a.time_stretch_rates), 3)
            self.assertEqual(a.time_stretch_rates[0],
                             a.time_stretch_rates[1]) # sync check
            self.assertEqual(len(a.pitch_shift_rates), 3)
            self.assertEqual(len(a.scale_amount_list), 3)
            self.assertEqual(len(a.scale_fraction_list), 3)

        # load journal
        with open(spec.journal_path, 'r') as fp:
            journal = AugmentationsJournal.from_dict(json.load(fp))

        self.assertEqual(journal.metadata_path, 'metadata.json')
        self.assertEqual(journal.spec.to_dict(), spec.to_dict())
        for j in journal.augmentation_journals:
            self.assertEqual(j.algorithm_out['score'],
                             j.algorithm_out['scoreStats']['max'])


    def test_frequency_augmentation_minscore(self):
        aug_dir = pathlib.Path(self.tmpdir.name) / 'augmentation_3'
        spec = AugmentationSpec(
            algorithm_name='FrequencyAugmentation',
            algorithm_params={
                'source_sample_rate': 8000,
                'target_sample_rate': 8000,
                'waveform_length': int(8000 * 0.5),

                'normalize': True,
                'scale_range': [0.1, 1.1],
                'scale_point_range': [1, 5],
                'time_stretch_range': [0.5, 1.5],
                'pitch_shift_range': [0.5, 1.5],

                'mixture_algorithm_name': 'CategoryMix',
                'mixture_algorithm_params': {
                    'mix_category_list': [
                        ['ct001'],
                        ['ct002', 'ct003']
                    ],
                    'include_other': False,
                },
                'trials_per_augmentation': 10,
                'separation_difficulty': 0.0,

            },
            seed=self.seed,
            augmentation_per_sample=2,
            sample_metadata_path=self.sample_dir/'e1'/'metadata.json',
            augmentation_metadata_path=aug_dir/'e1'/'metadata.json',
            journal_path=aug_dir/'e1'/'journal.json',
            jobs=None
        )
        spec.save_augmentation()

        # load metadata
        with open(spec.augmentation_metadata_path, 'r') as fp:
            augmentations = Augmentation.from_list(json.load(fp))

        self.assertEqual(len(augmentations), 2)
        for a in augmentations:
            self.assertEqual(a.sample_index, 0)
            self.assertEqual(a.source_sample_rate, 8000)
            self.assertEqual(a.target_sample_rate, 8000)
            self.assertEqual(a.waveform_length, int(8000 * 0.5))
            self.assertEqual(len(a.offsets), 3)
            self.assertEqual(a.offsets[0], a.offsets[1]) # sync check
            self.assertEqual(len(a.time_stretch_rates), 3)
            self.assertEqual(a.time_stretch_rates[0],
                             a.time_stretch_rates[1]) # sync check
            self.assertEqual(len(a.pitch_shift_rates), 3)
            self.assertEqual(len(a.scale_amount_list), 3)
            self.assertEqual(len(a.scale_fraction_list), 3)

        # load journal
        with open(spec.journal_path, 'r') as fp:
            journal = AugmentationsJournal.from_dict(json.load(fp))

        self.assertEqual(journal.metadata_path, 'metadata.json')
        self.assertEqual(journal.spec.to_dict(), spec.to_dict())
        for j in journal.augmentation_journals:
            self.assertEqual(j.algorithm_out['score'],
                             j.algorithm_out['scoreStats']['min'])

    def test_frequency_augmentation_maxscore(self):
        aug_dir = pathlib.Path(self.tmpdir.name) / 'augmentation_5'
        spec = AugmentationSpec(
            algorithm_name='FrequencyAugmentation',
            algorithm_params={
                'source_sample_rate': 8000,
                'target_sample_rate': 8000,
                'waveform_length': int(8000 * 0.5),

                'normalize': True,
                'scale_range': [0.1, 1.1],
                'scale_point_range': [1, 5],
                'time_stretch_range': [0.5, 1.5],
                'pitch_shift_range': [0.5, 1.5],

                'mixture_algorithm_name': 'CategoryMix',
                'mixture_algorithm_params': {
                    'mix_category_list': [
                        ['ct001'],
                        ['ct002', 'ct003']
                    ],
                    'include_other': False,
                },
                'trials_per_augmentation': 10,
                'separation_difficulty': 1.0,

            },
            seed=self.seed,
            augmentation_per_sample=2,
            sample_metadata_path=self.sample_dir/'e1'/'metadata.json',
            augmentation_metadata_path=aug_dir/'e1'/'metadata.json',
            journal_path=aug_dir/'e1'/'journal.json',
            jobs=None
        )
        spec.save_augmentation()

        # load metadata
        with open(spec.augmentation_metadata_path, 'r') as fp:
            augmentations = Augmentation.from_list(json.load(fp))

        self.assertEqual(len(augmentations), 2)
        for a in augmentations:
            self.assertEqual(a.sample_index, 0)
            self.assertEqual(a.source_sample_rate, 8000)
            self.assertEqual(a.target_sample_rate, 8000)
            self.assertEqual(a.waveform_length, int(8000 * 0.5))
            self.assertEqual(len(a.offsets), 3)
            self.assertEqual(a.offsets[0], a.offsets[1]) # sync check
            self.assertEqual(len(a.time_stretch_rates), 3)
            self.assertEqual(a.time_stretch_rates[0],
                             a.time_stretch_rates[1]) # sync check
            self.assertEqual(len(a.pitch_shift_rates), 3)
            self.assertEqual(len(a.scale_amount_list), 3)
            self.assertEqual(len(a.scale_fraction_list), 3)

        # load journal
        with open(spec.journal_path, 'r') as fp:
            journal = AugmentationsJournal.from_dict(json.load(fp))

        self.assertEqual(journal.metadata_path, 'metadata.json')
        self.assertEqual(journal.spec.to_dict(), spec.to_dict())
        for j in journal.augmentation_journals:
            self.assertEqual(j.algorithm_out['score'],
                             j.algorithm_out['scoreStats']['max'])

