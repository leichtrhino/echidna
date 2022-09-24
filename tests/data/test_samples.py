
import unittest
import os
import pathlib
import tempfile
import json
import math
import torch
import torchaudio

from collections import Counter

from echidna.data import samples
from echidna.data import datanodes

from .utils import prepare_datasources

class TestSamples(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.seed = 140343
        self.samplespecs = prepare_datasources(self.tmpdir, self.seed)
        self.sample_dir = pathlib.Path(self.tmpdir.name) / 'samples'

    def tearDown(self):
        self.tmpdir.cleanup()

    def _test_sample(self,
                    spec_key,
                    expect_categories,
                    expect_tracks,
                    expect_folds,
                    ):
        samplespec = self.samplespecs[spec_key]
        samplespec.save_samples()
        # check directory structure
        metadata_path = self.sample_dir / spec_key.lower() / 'metadata.json'
        journal_path = self.sample_dir / spec_key.lower() / 'journal.json'
        sample_dir = self.sample_dir / spec_key.lower()
        self.assertTrue(os.path.isfile(metadata_path))
        self.assertTrue(os.path.isfile(journal_path))
        self.assertTrue(os.path.isdir(sample_dir))

        # load metadata and data
        with open(metadata_path, 'r') as fp:
            datanode = datanodes.DataNode.from_dict(
                json.load(fp),
                context={'rel_path': os.path.dirname(metadata_path)}
            )

        self.assertEqual(len(datanode), 1)
        for data, metadata_list in datanode:
            metadata = metadata_list[0]
            for c_i, c in enumerate(metadata.channels):
                self.assertEqual(c.path, os.path.join('000', f'000.{c_i:02d}.pth'))
                self.assertIn(c.category, expect_categories)
                if expect_tracks is not None:
                    self.assertIn(c.track, expect_tracks)
                else:
                    self.assertEqual(c.track, None)
                if expect_folds is not None:
                    self.assertIn(c.fold, expect_folds)
                else:
                    self.assertEqual(c.fold, None)
                self.assertEqual(c.sample_rate, 8000)

                data = torch.load(sample_dir / c.path)
                self.assertTrue(type(data), dict)
                self.assertIn('wave', data)
                self.assertEqual(data['wave'].shape, (12000,))
                self.assertIn('sheet', data)

        # load journals
        with open(journal_path, 'r') as fp:
            journal = samples.SampleSetJournal.from_dict(json.load(fp))
        self.assertEqual(journal.metadata_path, 'metadata.json')
        self.assertEqual(journal.spec.to_dict(), samplespec.to_dict())
        self.assertEqual(journal.seed, self.seed)
        self.assertEqual(len(journal.sample_journals), 1)
        for j, (data, metadata_list) in \
            zip(journal.sample_journals, datanode):
            metadata = metadata_list[0]
            self.assertEqual(j.sample.to_dict(), metadata.to_dict())
            # find datasources from id
            for c in j.channel_journals:
                datasources = [
                    next(e for e in journal.spec.datasources if e.id == ds.id)
                    for ds in c.datasources
                ]
                self.assertEqual(c.channel.category, datasources[0].category)
                self.assertEqual(c.channel.track, datasources[0].track)
                self.assertEqual(c.channel.fold, datasources[0].fold)
                self.assertEqual(c.channel.category, datasources[0].category)
                self.assertEqual(
                    len(c.datasources),
                    1 if c.channel.category == 'ct001' else
                    2 if c.channel.category == 'ct002' else None
                )

    def test_sample_a1(self):
        self._test_sample(
            spec_key='A1',
            expect_categories=['ct001', 'ct002'],
            expect_tracks=None,
            expect_folds=None,
        )


    def test_sample_b1(self):
        self._test_sample(
            spec_key='B1',
            expect_categories=['ct001'],
            expect_tracks=None,
            expect_folds=['fl001'],
        )

    def test_sample_c2(self):
        self._test_sample(
            spec_key='C2',
            expect_categories=['ct001', 'ct002'],
            expect_tracks=[None, 'tk001', 'tk002'],
            expect_folds=None,
        )
