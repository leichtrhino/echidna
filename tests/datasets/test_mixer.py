
import unittest
import pathlib
import tempfile
import math
import itertools
import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt

from echidna.datasets.mixer import Mixer, FrozenMix
from echidna.datasets.mixer import CategoryMix
from echidna.datasets.mixer import freeze_mix, collate_mix
from echidna.datasets.sampler import Sampler, FrozenSamples

from .source_list import source_list_a, source_list_b

class TestMix(unittest.TestCase):
    def test_category_mix(self):
        mix_category_list = ['dog', ['rooster', 'cow']]
        output_other = False
        mix = CategoryMix(mix_category_list, output_other)

        data_in = {
            'waveform': torch.arange(1, 4).reshape(3, 1),
            'midi': None,
        }
        metadata_in = [
            {'category': 'dog'},
            {'category': 'rooster'},
            {'category': 'cow'},
        ]
        mix_index = mix.mix_index(data=data_in, metadata=metadata_in)
        self.assertEqual(
            mix_index,
            [((0,), (1,)), ((0,), (2,)), ((0,), (1, 2))]
        )

    def test_mix(self):
        duration = 4.
        sampler = Sampler(
            source_list_b,
            sr=16000,
            duration=duration,
            source_categories=['dog', 'rooster', 'cow'],
            splits=None,
        ).time_stretch_range((0.8, 1.2))
        mixer = Mixer(sampler, CategoryMix(mix_category_list=[['dog'],]))

        data, sources = next(iter(mixer))
        waveform, midi = data['waveform'], data['midi']

        # test data
        self.assertEqual(waveform.shape, torch.Size((2, int(16000*duration))))

        # test metadata
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0][0]['category'], 'dog')
        self.assertIn(sources[1][0]['category'], ['rooster', 'cow'])

    def test_freeze(self):
        sampler = Sampler(
            source_list_b,
            sr=16000,
            duration=5.,
            source_categories=['dog', 'rooster', 'cow'],
            splits=None,
        )
        mixer = CategoryMix(mix_category_list=[['dog'],])

        with tempfile.TemporaryDirectory() as tmpdirname:
            sampler.freeze(tmpdirname, 5, 4)
            sampler = FrozenSamples(tmpdirname)
            freeze_mix(sampler, mixer, 'samplemix')

            # 1. no metadata
            fmix = FrozenMix(sampler, 'samplemix')
            self.assertEqual(len(fmix), 3 * 5)

            loader = torch.utils.data.DataLoader(
                fmix, 2, collate_fn=collate_mix
            )
            d, s = next(iter(loader))
            waveform, midi = d['waveform'], d['midi']
            self.assertEqual(
                waveform.shape, torch.Size((2, 2, int(16000*5.))))

            self.assertEqual(len(s), 2)
            self.assertEqual(len(s[0]), 2)
            self.assertEqual(
                s[0][0][0]['category'], 'dog')
            self.assertIn(
                s[0][1][0]['category'], ['rooster', 'cow'])

