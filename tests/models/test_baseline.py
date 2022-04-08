
import unittest
import torch

from echidna.models import baseline as bl

class TestBaselineModels(unittest.TestCase):
    def test_baseline_encoder(self):
        encoder = bl.BaselineEncoder(in_channel=1,
                                     n_fft=128,
                                     n_lstm=2,
                                     lstm_channel=60,
                                     hop_length=32)

        x = torch.rand(8, 1, 16000)
        embd, X = encoder(x)
        F = encoder.forward_feature_size()
        T = encoder.forward_length(16000)
        self.assertEqual(embd.shape, (8, F, T))
        self.assertEqual(X.shape, (8, 1, 2, 128 // 2, T))

    def test_baseline_decoder(self):
        decoder = bl.BaselineDecoder(in_channel=1,
                                     out_channel=2,
                                     lstm_channel=60,
                                     n_fft=128,
                                     hop_length=32,
                                     magbook_size=1,
                                     phasebook_size=1,
                                     output_residual=False)

        embd = torch.rand(8, 60, 128)
        X = torch.rand(8, 1, 2, 128 // 2, 128)
        y = decoder(embd, X)
        T = decoder.forward_length(128)
        self.assertEqual(y.shape, (8, 2, T))

    def test_baseline_decoder_shared_masks(self):
        # test for shared masks
        decoder = bl.BaselineDecoder(in_channel=1,
                                     out_channel=2,
                                     lstm_channel=60,
                                     n_fft=128,
                                     hop_length=32,
                                     magbook_size=1,
                                     phasebook_size=1,
                                     output_residual=True)

        embd = torch.rand(8, 60, 128)
        X = torch.rand(8, 1, 2, 128 // 2, 128)
        y = decoder(embd, X)
        T = decoder.forward_length(128)
        self.assertEqual(y.shape, (8, 2, T))

    def test_baseline_decoder_multi_input(self):
        # test for multi input
        decoder = bl.BaselineDecoder(in_channel=2,
                                     out_channel=3,
                                     lstm_channel=60,
                                     n_fft=128,
                                     hop_length=32,
                                     magbook_size=1,
                                     phasebook_size=1,
                                     output_residual=False)

        embd = torch.rand(8, 60, 128)
        X = torch.rand(8, 1, 2, 128 // 2, 128)
        y = decoder(embd, X)
        T = decoder.forward_length(128)
        self.assertEqual(y.shape, (8, 3, 2, T))

    def test_baseline_decoder_multi_input_shared_mask(self):
        # test for multi input and shared mask
        decoder = bl.BaselineDecoder(in_channel=2,
                                     out_channel=4,
                                     lstm_channel=60,
                                     n_fft=128,
                                     hop_length=32,
                                     magbook_size=1,
                                     phasebook_size=1,
                                     output_residual=True)

        embd = torch.rand(8, 60, 128)
        X = torch.rand(8, 1, 2, 128 // 2, 128)
        y = decoder(embd, X)
        T = decoder.forward_length(128)
        self.assertEqual(y.shape, (8, 4, 2, T))

