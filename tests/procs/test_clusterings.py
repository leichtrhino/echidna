
import unittest
import tempfile
import pathlib
import json
import torch
import torchaudio

from echidna.models.models import InitialModel
from echidna.procs.clusterings import ClusteringSpec, ClusteringJournal
from .utils import get_encdec_model, get_chimera_model

class TestClustering(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        input = pathlib.Path(self.tmpdir.name) / 'input.wav'
        torchaudio.save(str(input), torch.zeros(1, 16000), sample_rate=8000)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _test_clustering(self,
                         name,
                         with_windowing=False,
                         with_journal_and_log=True,
                         use_chimera=True):
        if with_journal_and_log:
            journal_pattern = str(pathlib.Path(self.tmpdir.name)
                                  / name / 'journal_1.json')
            log_pattern = str(pathlib.Path(self.tmpdir.name)
                              / name / 'log_1.txt')
        else:
            journal_pattern = None
            log_pattern = None
        output = [
            str(pathlib.Path(self.tmpdir.name)/name/'output_1_1.wav'),
            str(pathlib.Path(self.tmpdir.name)/name/'output_1_2.wav'),
        ]

        spec = ClusteringSpec(
            # model
            model=get_encdec_model() if not use_chimera
            else get_chimera_model(),
            # input/output
            input=str(pathlib.Path(self.tmpdir.name)/'input.wav'),
            output=output,
            journal_pattern=journal_pattern,
            log_pattern=log_pattern,
            log_level='INFO',
            # clustering quality parameters
            sample_rate=8000,
            duration=0.5 if with_windowing else None,
            overlap=0.75 if with_windowing else None,
            n_fft=512,
            hop_length=128,
            # clustering process parameters
            batch_size=2 if with_windowing else None,
            device='cpu',
        )
        spec.cluster()

        # check output
        for o in output:
            self.assertTrue(pathlib.Path(o).exists)

        # check journal and log
        if with_journal_and_log:
            with open(journal_pattern, 'r') as fp:
                journal = ClusteringJournal.from_dict(json.load(fp))
            self.assertEqual(journal.spec.to_dict(), spec.to_dict())
            self.assertTrue(pathlib.Path(log_pattern).exists)

    def test_clustering_window(self):
        self._test_clustering(name='window',
                              with_windowing=True)


    def test_clustering_nowindow(self):
        self._test_clustering(name='nowindow',
                              with_windowing=False)

    def test_clustering_without_journal(self):
        self._test_clustering(name='nojournal',
                              with_windowing=True,
                              with_journal_and_log=False)

    def test_clustering_encdec(self):
        is_error = False
        try:
            self._test_clustering(name='chimera',
                                  with_windowing=True,
                                  use_chimera=False)
        except:
            is_error = True
        self.assertTrue(is_error)

