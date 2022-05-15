import unittest
import math
import itertools
import torch

import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from chimerau.models import convtasnet as ctn

class TestConvTasNetModels(unittest.TestCase):
    def test_encoder(self):
        encoder = ctn.TrainableStftLayer(512, 128)
        x = torch.rand(8, 2, 16000)
        X = encoder(x)
        self.assertEqual(len(X.shape), 4)
        self.assertEqual(X.shape[:-2], (8, 2))
        self.assertEqual(X.shape[-2], 2*(512 // 2 + 1))
        self.assertEqual(X.shape[-1], (16000 + 2 * 128*2 - 512) // 128 + 1)
        self.assertEqual(encoder.forward_length(16000), X.shape[-1])
        self.assertEqual(encoder.reverse_length(X.shape[-1]), 16000)

    def test_decoder(self):
        decoder = ctn.TrainableIstftLayer(512, 128)
        X = torch.rand(8, 2, 512+2, 126)
        x = decoder(X)
        self.assertEqual(len(x.shape), 3)
        self.assertEqual(x.shape[:-1], (8, 2))
        self.assertEqual(x.shape[-1], 128 * (126-1) - 2*128*2 + 512)
        self.assertEqual(decoder.forward_length(126), x.shape[-1])
        self.assertEqual(decoder.reverse_length(x.shape[-1]), 126)

    def test_conv_block(self):
        block = ctn.ConvBlock(io_channel=14,
                              block_channel=12,
                              skipconnection_channel=16,
                              kernel_size=9,
                              dilation=8)
        x = torch.rand(8, 14, 20)
        sc, o = block(x)
        self.assertEqual(sc.shape, (8, 16, 20))
        self.assertEqual(o.shape, (8, 14, 20))
        self.assertEqual(block.forward_length(20), sc.shape[-1])
        self.assertEqual(block.forward_length(20), o.shape[-1])
        self.assertEqual(block.reverse_length(20), sc.shape[-1])
        self.assertEqual(block.reverse_length(20), o.shape[-1])

    def test_conv_tasnet(self):
        ctasnet = ctn.ConvTasNet(in_channel=1,
                                 out_channel=2,
                                 feature_channel=128,
                                 block_channel=14,
                                 bottleneck_channel=12,
                                 skipconnection_channel=16,
                                 kernel_size=9,
                                 depth=3,
                                 repeats=2)

        x = torch.rand(8, 1, 16000)
        signals = ctasnet(x)
        self.assertEqual(signals.shape, (8, 2, 16000))

        signals, embd = ctasnet(x, return_embd=True)
        F = ctasnet.forward_embd_feature()
        T = ctasnet.forward_embd_length(16000)
        self.assertEqual(signals.shape, (8, 2, 16000))
        self.assertEqual(embd.shape, (8, F, T))

        # test for shared masks
        ctasnet = ctn.ConvTasNet(in_channel=1,
                                 out_channel=2,
                                 feature_channel=130,
                                 block_channel=14,
                                 bottleneck_channel=12,
                                 skipconnection_channel=16,
                                 kernel_size=9,
                                 depth=3,
                                 repeats=2,
                                 mask_each=False)
        signals = ctasnet(x)
        self.assertEqual(signals.shape, (8, 2, 16000))

        # test for multi input
        ctasnet = ctn.ConvTasNet(in_channel=2,
                                 out_channel=3,
                                 feature_channel=130,
                                 block_channel=14,
                                 bottleneck_channel=12,
                                 skipconnection_channel=16,
                                 kernel_size=9,
                                 depth=3,
                                 repeats=2)

        x = torch.rand(8, 2, 16000)
        signals = ctasnet(x)
        self.assertEqual(signals.shape, (8, 3, 2, 16000))

        signals, embd = ctasnet(x, return_embd=True)
        F = ctasnet.forward_embd_feature()
        T = ctasnet.forward_embd_length(16000)
        self.assertEqual(signals.shape, (8, 3, 2, 16000))
        self.assertEqual(embd.shape, (8, F, T))

        # test for codebook>1
        ctasnet = ctn.ConvTasNet(in_channel=1,
                                 out_channel=2,
                                 feature_channel=130,
                                 block_channel=14,
                                 bottleneck_channel=12,
                                 skipconnection_channel=16,
                                 kernel_size=9,
                                 depth=3,
                                 repeats=2,
                                 mask_each=True,
                                 magbook_size=3,
                                 phasebook_size=8,)
        x = torch.rand(8, 1, 16000)
        signals = ctasnet(x)
        self.assertEqual(signals.shape, (8, 2, 16000))

        ctasnet = ctn.ConvTasNet(in_channel=1,
                                 out_channel=2,
                                 feature_channel=130,
                                 block_channel=14,
                                 bottleneck_channel=12,
                                 skipconnection_channel=16,
                                 kernel_size=9,
                                 depth=3,
                                 repeats=2,
                                 mask_each=False,
                                 magbook_size=3,
                                 phasebook_size=8,)
        signals = ctasnet(x)
        self.assertEqual(signals.shape, (8, 2, 16000))



    def test_chimera_conv_tasnet(self):
        cctasnet = ctn.ChimeraConvTasNet(in_channel=1,
                                         out_channel=2,
                                         feature_channel=128,
                                         block_channel=14,
                                         bottleneck_channel=12,
                                         skipconnection_channel=16,
                                         embd_feature=128,
                                         embd_dim=20,
                                         kernel_size=9,
                                         depth=3,
                                         repeats=2)

        x = torch.rand(8, 1, 16000)
        signals, embd = cctasnet(x)
        self.assertEqual(signals.shape, (8, 2, 16000))
        F = cctasnet.forward_embd_feature()
        T = cctasnet.forward_embd_length(16000)
        self.assertEqual(embd.shape, (8, F, T, 20))

        cctasnet = ctn.ChimeraConvTasNet(in_channel=1,
                                         out_channel=2,
                                         feature_channel=128,
                                         block_channel=14,
                                         bottleneck_channel=12,
                                         skipconnection_channel=16,
                                         embd_feature=128,
                                         embd_dim=20,
                                         kernel_size=9,
                                         depth=3,
                                         repeats=2,
                                         magbook_size=3,
                                         phasebook_size=8)

        x = torch.rand(8, 1, 16000)
        signals, embd = cctasnet(x)
        self.assertEqual(signals.shape, (8, 2, 16000))
        F = cctasnet.forward_embd_feature()
        T = cctasnet.forward_embd_length(16000)
        self.assertEqual(embd.shape, (8, F, T, 20))

if __name__ == '__main__':
    unittest.main()
