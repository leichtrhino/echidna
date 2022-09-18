
import unittest
import pathlib
import tempfile
import json

from echidna.data.datanodes import DataNode
from echidna.data.difficulties import (
    MetricSpec, MetricSetJournal, EntropyDifficulty, FrequencyDifficulty,
)

from .utils import prepare_datasources

class TestDifficulties(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.seed = 140343
        self.samplespecs = prepare_datasources(self.tmpdir, self.seed)
        self.samplespec = self.samplespecs['A1']
        self.samplespec.save_samples()
        self.sample_dir = pathlib.Path(self.tmpdir.name) / 'samples'

    def tearDown(self):
        self.tmpdir.cleanup()

    def _test_metric_difficulty(self,
                                metric_class,
                                additional_params):
        metric_name = 'metric'
        output_metadata_path = str(
            self.sample_dir/'a1'/'metric_metadata.json')
        journal_path = str(
            self.sample_dir/'a1'/'metric_journal.json')
        log_path = str(
            self.sample_dir/'a1'/'metric_log.txt')

        spec = metric_class(
            metric_name=metric_name,
            input_metadata_path=self.samplespec.metadata_path,
            output_metadata_path=output_metadata_path,
            seed=1410343,
            journal_path=journal_path,
            log_path=log_path,
            log_level='INFO',
            jobs=None,
            device='cpu',
            **additional_params,
        )
        spec.put_metrics()

        # load metadata
        with open(spec.output_metadata_path, 'r') as fp:
            datanode = DataNode.from_dict(
                obj=json.load(fp),
                context={
                    'rel_path': pathlib.Path(spec.output_metadata_path).parent,
                }
            )

        leafnodes = datanode.list_leaf_node()
        for n in leafnodes:
            self.assertGreater(n.get_metric(metric_name), -1000)

        # load journal
        with open(spec.journal_path, 'r') as fp:
            journal = MetricSetJournal.from_dict(json.load(fp))

        self.assertEqual(journal.metadata_path, 'metric_metadata.json')
        self.assertEqual(journal.spec.to_dict(), spec.to_dict())

    def test_entropy_difficulty(self):
        self._test_metric_difficulty(
            EntropyDifficulty,
            {
                'n_fft': 2048,
                'win_length': 2048,
                'hop_length': 512,
            }
        )

    def test_frequency_difficulty(self):
        self._test_metric_difficulty(
            FrequencyDifficulty,
            {
                'sample_rate': 8000,
                'win_length': 2048,
            }
        )

