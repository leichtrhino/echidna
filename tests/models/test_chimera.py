
import unittest
import torch

from echidna.models import baseline as bl
from echidna.models.chimera import ChimeraNet

class TestChimeraNetModels(unittest.TestCase):
    def test_baseline(self):
        encoder_class = bl.BaselineEncoder
        decoder_class = bl.BaselineDecoder
        hyperparameters = dict(
            in_channel=1,
            out_channel=2,
            lstm_channel=20,
            n_fft=128,
            hop_length=32,
            n_lstm=2,
            mask_each=False,
        )
        chimera = ChimeraNet(encoder_class,
                             decoder_class,
                             hyperparameters,
                             embd_feature=64,
                             embd_dim=16)

        x = torch.rand(8, 1, 16000)
        y, embd = chimera(x)
        self.assertEqual(
            y.shape,
            (8, 2, chimera.forward_length(16000))
        )
        self.assertEqual(
            embd.shape,
            (8, 64, chimera.forward_embd_length(16000), 16)
        )
