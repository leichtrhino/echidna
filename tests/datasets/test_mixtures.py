
import unittest
import pathlib
import tempfile
import json

from echidna.datasets.mixtures import (
    Mixture, MixturesJournal, MixtureSpec
)
from echidna.datasets.samples import SampleSpec

from .utils import prepare_datasources

class TestMixtures(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.seed = 140343
        self.samplespecs = prepare_datasources(self.tmpdir, self.seed)
        self.samplespec = self.samplespecs['D1']
        self.samplespec.save_samples()
        self.sample_dir = pathlib.Path(self.tmpdir.name) / 'samples'

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_category_mix_include_other(self):
        mix_dir = pathlib.Path(self.tmpdir.name) / 'mixtures_1'
        spec = MixtureSpec(
            algorithm_name='CategoryMix',
            algorithm_params={
                'mix_category_list': [['ct001'], ['ct002', 'ct003']],
                'include_other': True,
            },
            seed=self.seed,
            mix_per_sample=2,
            sample_metadata_path=self.sample_dir/'d1'/'metadata.json',
            mixture_metadata_path=mix_dir/'d1'/'metadata.json',
            journal_path=mix_dir/'d1'/'journal.json',
            jobs=None
        )
        spec.save_mixture()

        # load metadata
        with open(spec.mixture_metadata_path, 'r') as fp:
            mixtures = Mixture.from_list(json.load(fp))

        self.assertEqual(len(mixtures), 2)
        for m in mixtures:
            self.assertEqual(m.sample_index, 0)
            self.assertEqual(m.mixture_indices,
                             [
                                 [[0], [1], []],
                                 [[0], [2], []],
                                 [[0], [1, 2], []],
                             ])

        # load journal
        with open(spec.journal_path, 'r') as fp:
            journal = MixturesJournal.from_dict(json.load(fp))

        self.assertEqual(journal.metadata_path, 'metadata.json')
        self.assertEqual(journal.spec.to_dict(), spec.to_dict())

    def test_category_mix_exclude_other(self):
        mix_dir = pathlib.Path(self.tmpdir.name) / 'mixtures_2'
        spec = MixtureSpec(
            algorithm_name='CategoryMix',
            algorithm_params={
                'mix_category_list': [['ct001'], ['ct002', 'ct003']],
                'include_other': False,
            },
            seed=self.seed,
            mix_per_sample=2,
            sample_metadata_path=self.sample_dir/'d1'/'metadata.json',
            mixture_metadata_path=mix_dir/'d1'/'metadata.json',
            journal_path=mix_dir/'d1'/'journal.json',
            jobs=None
        )
        spec.save_mixture()

        # load metadata
        with open(spec.mixture_metadata_path, 'r') as fp:
            mixtures = Mixture.from_list(json.load(fp))

        self.assertEqual(len(mixtures), 2)
        for m in mixtures:
            self.assertEqual(m.sample_index, 0)
            self.assertEqual(m.mixture_indices,
                             [
                                 [[0], [1]],
                                 [[0], [2]],
                                 [[0], [1, 2]],
                             ])

        # load journal
        with open(spec.journal_path, 'r') as fp:
            journal = MixturesJournal.from_dict(json.load(fp))

        self.assertEqual(journal.metadata_path, 'metadata.json')
        self.assertEqual(journal.spec.to_dict(), spec.to_dict())

