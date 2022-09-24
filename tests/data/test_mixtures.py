
import os
import unittest
import pathlib
import tempfile
import json

from echidna.data.datanodes import DataNode
from echidna.data.mixtures import (
    MixNode, MixSetSpec, MixSetJournal
)

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

    def test_category_mix(self):
        mix_dir = pathlib.Path(self.tmpdir.name) / 'mixtures_1'
        spec = MixSetSpec(
            input_metadata_path=self.sample_dir/'d1'/'metadata.json',
            output_metadata_path=mix_dir/'d1'/'metadata.json',
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
            mix_per_parent=None,
            seed=self.seed,
            journal_path=mix_dir/'d1'/'journal.json',
            log_path=mix_dir/'d1'/'log.txt',
            log_level='INFO',
            jobs=None
        )
        spec.save_mixture()

        # load metadata
        with open(spec.output_metadata_path, 'r') as fp:
            rootnode = DataNode.from_dict(
                json.load(fp),
                context={
                    'rel_path': os.path.dirname(spec.input_metadata_path)
                }
            )

        self.assertEqual(len(rootnode), 3)

        true_mix_index = set([
            ((0,), (1,)),
            ((0,), (2,)),
            ((0,), (1, 2)),
        ])
        test_mix_index = set([
            tuple(tuple(mi) for mi in m.mix_index)
            for m in rootnode.list_leaf_node()
        ])
        self.assertEqual(test_mix_index, true_mix_index)

        # load data
        for data, metadata in rootnode:
            self.assertEqual(len(data), 2)

        # load journal
        with open(spec.journal_path, 'r') as fp:
            journal = MixSetJournal.from_dict(json.load(fp))

        self.assertEqual(journal.metadata_path, 'metadata.json')
        self.assertEqual(journal.spec.to_dict(), spec.to_dict())

