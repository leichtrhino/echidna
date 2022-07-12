
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

from .utils import prepare_datasources

class TestSamples(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.seed = 140343
        self.samplespecs = prepare_datasources(self.tmpdir, self.seed)
        self.sample_dir = pathlib.Path(self.tmpdir.name) / 'samples'

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_sample_a1(self):
        samplespec = self.samplespecs['A1']
        samplespec.save_samples()
        # check directory structure
        metadata_path = self.sample_dir / 'a1' / 'metadata.json'
        journal_path = self.sample_dir / 'a1' / 'journal.json'
        sample_path = self.sample_dir / 'a1' / '000' / '000.pth'
        self.assertTrue(os.path.isfile(metadata_path))
        self.assertTrue(os.path.isfile(journal_path))
        self.assertTrue(os.path.isfile(sample_path))

        # load metadata
        with open(metadata_path, 'r') as fp:
            samples_ = samples.Sample.from_list(json.load(fp))
        self.assertEqual(len(samples_), 1)
        for s in samples_:
            self.assertEqual(s.path, os.path.join('000', '000.pth'))
            self.assertEqual(s.categories, ['ct001', 'ct002'])
            self.assertEqual(s.tracks, [None, None])
            self.assertEqual(s.folds, [None, None])
            self.assertEqual(s.sample_rate, 8000)

        # load journals
        with open(journal_path, 'r') as fp:
            journal = samples.SamplesJournal.from_dict(json.load(fp))
        self.assertEqual(journal.metadata_path, 'metadata.json')
        self.assertEqual(journal.spec.to_dict(), samplespec.to_dict())
        self.assertEqual(journal.seed, self.seed)
        self.assertEqual(len(journal.sample_journals), 1)
        for j, s in zip(journal.sample_journals, samples_):
            self.assertEqual(j.sample.to_dict(), s.to_dict())
            self.assertEqual(j.length, 12000)
            # find datasources from id
            datasources = [
                [
                    next(e for e in journal.spec.datasources
                         if e.id == dsid)
                    for dsid in dsids
                ]
                for dsids in j.datasources
            ]
            self.assertEqual(
                [dss[0].category for dss in datasources],
                s.categories
            )
            self.assertEqual(
                [dss[0].track for dss in datasources],
                s.tracks
            )
            self.assertEqual(
                [dss[0].fold for dss in datasources],
                s.folds
            )
            self.assertEqual(
                [len(dss) for dss in datasources],
                [1 if c == 'ct001' else 2 if c == 'ct002' else None
                 for c in s.categories]
            )

        # load data
        data = torch.load(sample_path)
        self.assertTrue(type(data), dict)
        self.assertIn('waves', data)
        self.assertEqual(data['waves'].shape, (2, 12000))
        self.assertIn('sheets', data)

    def test_sample_a2(self):
        samplespec = self.samplespecs['A2']
        samplespec.save_samples()
        # check directory structure
        metadata_path = self.sample_dir / 'a2' / 'metadata.json'
        journal_path = self.sample_dir / 'a2' / 'journal.json'
        sample_path = self.sample_dir / 'a2' / '000' / '000.pth'
        self.assertTrue(os.path.isfile(metadata_path))
        self.assertTrue(os.path.isfile(journal_path))
        self.assertTrue(os.path.isfile(sample_path))

        # load metadata
        with open(metadata_path, 'r') as fp:
            samples_ = samples.Sample.from_list(json.load(fp))
        self.assertEqual(len(samples_), 1)
        for s in samples_:
            self.assertEqual(s.path, os.path.join('000', '000.pth'))
            self.assertEqual(s.categories, ['ct001', 'ct001', 'ct002'])
            self.assertEqual(s.tracks, [None, None, None])
            self.assertEqual(s.folds, [None, None, None])
            self.assertEqual(s.sample_rate, 8000)

        # load journals
        with open(journal_path, 'r') as fp:
            journal = samples.SamplesJournal.from_dict(json.load(fp))
        self.assertEqual(journal.metadata_path, 'metadata.json')
        self.assertEqual(journal.spec.to_dict(), samplespec.to_dict())
        self.assertEqual(journal.seed, self.seed)
        self.assertEqual(len(journal.sample_journals), 1)
        for j, s in zip(journal.sample_journals, samples_):
            self.assertEqual(j.sample.to_dict(), s.to_dict())
            self.assertEqual(j.length, 12000)
            # find datasources from id
            datasources = [
                [
                    next(e for e in journal.spec.datasources
                         if e.id == dsid)
                    for dsid in dsids
                ]
                for dsids in j.datasources
            ]
            self.assertEqual(
                [dss[0].category for dss in datasources],
                s.categories
            )
            self.assertEqual(
                [dss[0].track for dss in datasources],
                s.tracks
            )
            self.assertEqual(
                [dss[0].fold for dss in datasources],
                s.folds
            )
            self.assertEqual(
                [len(dss) for dss in datasources],
                [1 if c == 'ct001' else 2 if c == 'ct002' else None
                 for c in s.categories]
            )

        # load data
        data = torch.load(sample_path)
        self.assertTrue(type(data), dict)
        self.assertIn('waves', data)
        self.assertEqual(data['waves'].shape, (3, 12000))
        self.assertIn('sheets', data)

    def test_sample_b1(self):
        samplespec = self.samplespecs['B1']
        samplespec.save_samples()
        # check directory structure
        metadata_path = self.sample_dir / 'b1' / 'metadata.json'
        journal_path = self.sample_dir / 'b1' / 'journal.json'
        sample_path = self.sample_dir / 'b1' / '000' / '000.pth'
        self.assertTrue(os.path.isfile(metadata_path))
        self.assertTrue(os.path.isfile(journal_path))
        self.assertTrue(os.path.isfile(sample_path))

        # load metadata
        with open(metadata_path, 'r') as fp:
            samples_ = samples.Sample.from_list(json.load(fp))
        self.assertEqual(len(samples_), 1)
        for s in samples_:
            self.assertEqual(s.path, os.path.join('000', '000.pth'))
            self.assertEqual(s.categories, ['ct001'])
            self.assertEqual(s.tracks, [None])
            self.assertEqual(s.folds, ['fl001'])
            self.assertEqual(s.sample_rate, 8000)

        # load journals
        with open(journal_path, 'r') as fp:
            journal = samples.SamplesJournal.from_dict(json.load(fp))
        self.assertEqual(journal.metadata_path, 'metadata.json')
        self.assertEqual(journal.spec.to_dict(), samplespec.to_dict())
        self.assertEqual(journal.seed, self.seed)
        self.assertEqual(len(journal.sample_journals), 1)
        for j, s in zip(journal.sample_journals, samples_):
            self.assertEqual(j.sample.to_dict(), s.to_dict())
            self.assertEqual(j.length, 12000)
            # find datasources from id
            datasources = [
                [
                    next(e for e in journal.spec.datasources
                         if e.id == dsid)
                    for dsid in dsids
                ]
                for dsids in j.datasources
            ]
            self.assertEqual(
                [dss[0].category for dss in datasources],
                s.categories
            )
            self.assertEqual(
                [dss[0].track for dss in datasources],
                s.tracks
            )
            self.assertEqual(
                [dss[0].fold for dss in datasources],
                s.folds
            )
            self.assertEqual(
                [len(dss) for dss in datasources],
                [1 if c == 'ct001' else 2 if c == 'ct002' else None
                 for c in s.categories]
            )

        # load data
        data = torch.load(sample_path)
        self.assertTrue(type(data), dict)
        self.assertIn('waves', data)
        self.assertEqual(data['waves'].shape, (1, 12000))
        self.assertIn('sheets', data)

    def test_sample_c2(self):
        samplespec = self.samplespecs['C2']
        samplespec.save_samples()
        # check directory structure
        metadata_path = self.sample_dir / 'c2' / 'metadata.json'
        journal_path = self.sample_dir / 'c2' / 'journal.json'
        sample_path = self.sample_dir / 'c2' / '000' / '000.pth'
        self.assertTrue(os.path.isfile(metadata_path))
        self.assertTrue(os.path.isfile(journal_path))
        self.assertTrue(os.path.isfile(sample_path))

        # load metadata
        with open(metadata_path, 'r') as fp:
            samples_ = samples.Sample.from_list(json.load(fp))
        self.assertEqual(len(samples_), 1)
        for s in samples_:
            self.assertEqual(s.path, os.path.join('000', '000.pth'))
            self.assertEqual(s.categories, ['ct001', 'ct002'])
            if s.tracks[0] == 'tk001':
                self.assertEqual(s.tracks, ['tk001', None])
            elif s.tracks[0] == 'tk002':
                self.assertEqual(s.tracks, ['tk002', None])
            else:
                raise ValueError()
            self.assertEqual(s.folds, [None, None])
            self.assertEqual(s.sample_rate, 8000)

        # load journals
        with open(journal_path, 'r') as fp:
            journal = samples.SamplesJournal.from_dict(json.load(fp))
        self.assertEqual(journal.metadata_path, 'metadata.json')
        self.assertEqual(journal.spec.to_dict(), samplespec.to_dict())
        self.assertEqual(journal.seed, self.seed)
        self.assertEqual(len(journal.sample_journals), 1)
        for j, s in zip(journal.sample_journals, samples_):
            self.assertEqual(j.sample.to_dict(), s.to_dict())
            self.assertEqual(j.length, 12000)
            # find datasources from id
            datasources = [
                [
                    next(e for e in journal.spec.datasources
                         if e.id == dsid)
                    for dsid in dsids
                ]
                for dsids in j.datasources
            ]
            self.assertEqual(
                [dss[0].category for dss in datasources],
                s.categories
            )
            self.assertEqual(
                [dss[0].track for dss in datasources],
                s.tracks
            )
            self.assertEqual(
                [dss[0].fold for dss in datasources],
                s.folds
            )
            self.assertEqual(
                [len(dss) for dss in datasources],
                [1 if c == 'ct001' else 2 if c == 'ct002' else None
                 for c in s.categories]
            )

        # load data
        data = torch.load(sample_path)
        self.assertTrue(type(data), dict)
        self.assertIn('waves', data)
        self.assertEqual(data['waves'].shape, (2, 12000))
        self.assertIn('sheets', data)

        # test logs
        log_path = self.sample_dir / 'c2' / 'log.txt'
        with open(log_path, 'r') as fp:
            for l in fp:
                event_dict = json.loads(l[l.index(' ')+1:])
                self.assertIn(event_dict['type'],
                              {'start_sampling', 'made_sample',
                               'save_samples', 'save_samples_journal',
                               'finish_sampling'})
