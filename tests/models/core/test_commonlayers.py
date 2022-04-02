
import unittest
import torch

from echidna.models.core import commonlayers

class TestCommonLayers(unittest.TestCase):
    def test_encoder(self):
        # port from convtasnet
        encoder = commonlayers.TrainableSTFTLayer(512, 128)
        x = torch.rand(8, 2, 16000)
        X = encoder(x)
        self.assertEqual(len(X.shape), 5)
        self.assertEqual(X.shape[:-2], (8, 2, 2))
        self.assertEqual(X.shape[-3], 2)
        self.assertEqual(X.shape[-2], (512 // 2))
        self.assertEqual(X.shape[-1], (16000 + 2 * 128*2 - 512) // 128 + 1)
        self.assertEqual(encoder.forward_length(16000), X.shape[-1])
        self.assertEqual(encoder.reverse_length(X.shape[-1]), 16000)

    def test_decoder(self):
        # import from convtasnet
        decoder = commonlayers.TrainableISTFTLayer(512, 128)
        X = torch.rand(8, 2, 2, 512 // 2, 126)
        x = decoder(X)
        self.assertEqual(len(x.shape), 3)
        self.assertEqual(x.shape[:-1], (8, 2))
        self.assertEqual(x.shape[-1], 128 * (126-1) - 2*128*2 + 512)
        self.assertEqual(decoder.forward_length(126), x.shape[-1])
        self.assertEqual(decoder.reverse_length(x.shape[-1]), 126)

    def test_trainable_stft_layer(self):
        tstft = commonlayers.TrainableSTFTLayer(16, 4)
        tistft = commonlayers.TrainableISTFTLayer(16, 4)

        x = torch.rand(8, 10, 64)
        X = tstft(x)
        self.assertEqual(X.shape, (8, 10, 2, 16//2, 64//4+1))

        xhat = tistft(X)
        self.assertEqual(xhat.shape, (8, 10, 64))

        x = torch.sin(2*torch.arange(4096)/torch.pi)\
                 .unsqueeze(0).unsqueeze(0)
        tstft = commonlayers.TrainableSTFTLayer(1024, 256)
        tistft = commonlayers.TrainableISTFTLayer(1024, 256)
        Xhat = tstft(x)
        xhat = tistft(Xhat)

        stft = commonlayers.STFTLayer(1024, 256)
        istft = commonlayers.ISTFTLayer(1024, 256)
        X = stft(x)
        xrcv = istft(X)

    def test_sigmoid_mask(self):
        mask_layer = commonlayers.SigmoidMask(in_channel=60,
                                        out_channel=128,
                                        mask_num=4)
        x = torch.rand(8, 60, 1200)
        y = mask_layer(x)
        self.assertEqual(y.shape, (8, 4, 2, 128 // 2, 1200))

    def test_codebook_mask(self):
        mask_layer = commonlayers.CodebookMask(in_channel=60,
                                               out_channel=128,
                                               mask_num=4,
                                               magbook_size=3,
                                               phasebook_size=8)
        x = torch.rand(8, 60, 1200)
        y = mask_layer(x)
        self.assertEqual(y.shape, (8, 4, 2, 128 // 2, 1200))

