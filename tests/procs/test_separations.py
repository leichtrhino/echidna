
import unittest
import tempfile
import pathlib
import json
import torch
import torchaudio

from echidna.models.models import InitialModel
from echidna.procs.separations import SeparationSpec, SeparationJournal
from .utils import get_encdec_model, get_chimera_model

class TestSeparation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        input = pathlib.Path(self.tmpdir.name) / 'input.wav'
        torchaudio.save(str(input), torch.zeros(1, 16000), sample_rate=8000)
        input_2 = pathlib.Path(self.tmpdir.name) / 'input_2.wav'
        torchaudio.save(str(input), torch.zeros(2, 24240), sample_rate=8000)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _test_separation(self,
                         name,
                         with_windowing=False,
                         permutation_invariant=False,
                         with_journal_and_log=True,
                         use_chimera=False,
                         multi_channel=False,
                         ):
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
            str(pathlib.Path(self.tmpdir.name)/name/'output_1_3.wav'),
        ]

        spec = SeparationSpec(
            # model
            model=get_encdec_model() if not use_chimera
            else get_chimera_model(),
            # input/output
            input=str(pathlib.Path(self.tmpdir.name)/'input.wav'),
            output=output,
            journal_pattern=journal_pattern,
            log_pattern=log_pattern,
            log_level='INFO',
            # separation quality parameters
            sample_rate=8000,
            duration=0.5 if with_windowing else None,
            overlap=0.75 if with_windowing else None,
            permutation_invariant=permutation_invariant,
            # separation process parameters
            batch_size=2 if with_windowing else None,
            device='cpu',
        )
        spec.separate()

        # check output
        for o in output:
            self.assertTrue(pathlib.Path(o).exists)

        # check journal and log
        if with_journal_and_log:
            with open(journal_pattern, 'r') as fp:
                journal = SeparationJournal.from_dict(json.load(fp))
            self.assertEqual(journal.spec.to_dict(), spec.to_dict())
            self.assertTrue(pathlib.Path(log_pattern).exists)

    def test_separation_window(self):
        self._test_separation(name='window',
                              with_windowing=True)


    def test_separation_nowindow(self):
        self._test_separation(name='nowindow',
                              with_windowing=False)

    def test_separation_pit(self):
        self._test_separation(name='pit',
                              with_windowing=True,
                              permutation_invariant=True)

    def test_separation_without_journal(self):
        self._test_separation(name='nojournal',
                              with_windowing=True,
                              with_journal_and_log=False)

    def test_separation_chimera(self):
        self._test_separation(name='chimera',
                              with_windowing=True,
                              use_chimera=True)

    def test_separation_window_multichannel(self):
        self._test_separation(name='window_2',
                              with_windowing=True,
                              multi_channel=True)


    def test_separation_nowindow_multichannel(self):
        self._test_separation(name='nowindow_2',
                              with_windowing=False,
                              multi_channel=True)

