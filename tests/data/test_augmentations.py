
import os
import unittest
import pathlib
import tempfile
import json

from echidna.data.augmentations import (
    AugmentationNode,
    AugmentationSetJournal,
    AugmentationSpec,
)
from echidna.data.datanodes import DataNode

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
            source_sample_rate=8000,
            target_sample_rate=8000,
            waveform_length=int(8000 * 0.5),

            scale_range=[0.1, 1.1],
            scale_point_range=[1, 5],
            time_stretch_range=[0.5, 1.5],
            pitch_shift_range=[0.5, 1.5],

            input_metadata_path=self.sample_dir/'e1'/'metadata.json',
            output_metadata_path=aug_dir/'e1'/'metadata.json',
            seed=self.seed,
            augmentation_per_parent=2,
            journal_path=aug_dir/'e1'/'journal.json',
            log_path=aug_dir/'e1'/'log.txt',
            log_level='DEBUG',
            jobs=None
        )
        spec.save_augmentation()

        # load metadata
        with open(spec.output_metadata_path, 'r') as fp:
            rootnode = DataNode.from_dict(
                json.load(fp),
                context={
                    'rel_path': os.path.dirname(spec.input_metadata_path),
                }
            )
        self.assertEqual(len(rootnode), 2)

        for a in rootnode.list_leaf_node():
            self.assertEqual(a.source_sample_rate, 8000)
            self.assertEqual(a.target_sample_rate, 8000)
            self.assertEqual(a.waveform_length, int(8000 * 0.5))
            self.assertEqual(len(a.channel_augmentations), 3)
            self.assertEqual( # sync check
                a.channel_augmentations[0].offset,
                a.channel_augmentations[1].offset
            )
            self.assertEqual( # sync check
                a.channel_augmentations[0].time_stretch_rate,
                a.channel_augmentations[1].time_stretch_rate
            )

        # load data
        for data, metadata in rootnode:
            self.assertEqual(len(data), 3)
            for d in data:
                self.assertEqual(d['wave'].shape, (int(8000 * 0.5),))

        # load journal
        with open(spec.journal_path, 'r') as fp:
            journal = AugmentationSetJournal.from_dict(json.load(fp))

        self.assertEqual(journal.metadata_path, 'metadata.json')
        self.assertEqual(journal.spec.to_dict(), spec.to_dict())

        # load logs
        with open(spec.log_path, 'r') as fp:
            for l in fp:
                event_dict = json.loads(l[l.index(' '):])
                self.assertIn(event_dict['type'],
                              {'start_augmentation',
                               'made_augmentation',
                               'save_augmentations',
                               'save_augmentations_journal',
                               'finish_augmentation',})

