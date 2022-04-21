
import unittest
import torch

from echidna.models import convtasnet as ctn

class TestConvTasNetModels(unittest.TestCase):
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

    def test_convtasnetencoder(self):
        encoder = ctn.ConvTasNetEncoder(encoder_in_channel=1,
                                        feature_channel=128,
                                        block_channel=14,
                                        bottleneck_channel=12,
                                        skipconnection_channel=16,
                                        kernel_size=9,
                                        depth=3,
                                        repeats=2)

        target_length = 200
        input_length = encoder.reverse_length(target_length)
        output_length = encoder.forward_length(input_length)
        channel = encoder.forward_channel()
        self.assertEqual(channel, 16)

        x = torch.rand(8, input_length)
        embd, X = encoder(x)
        self.assertEqual(embd.shape, (8, channel, output_length))
        self.assertEqual(X.shape, (8, 1, 2, 128 // 2, output_length))

    def test_convtasnetencoder_multiinput(self):
        encoder = ctn.ConvTasNetEncoder(encoder_in_channel=2,
                                        feature_channel=128,
                                        block_channel=14,
                                        bottleneck_channel=12,
                                        skipconnection_channel=16,
                                        kernel_size=9,
                                        depth=3,
                                        repeats=2)

        target_length = 200
        input_length = encoder.reverse_length(target_length)
        output_length = encoder.forward_length(input_length)
        channel = encoder.forward_channel()
        self.assertEqual(channel, 16)

        x = torch.rand(8, 2, input_length)
        embd, X = encoder(x)
        self.assertEqual(embd.shape, (8, channel, output_length))
        self.assertEqual(X.shape, (8, 2, 2, 128 // 2, output_length))



    def test_convtasnetdecoder(self):
        encoder = ctn.ConvTasNetEncoder(encoder_in_channel=1,
                                        feature_channel=128,
                                        block_channel=14,
                                        bottleneck_channel=12,
                                        skipconnection_channel=16,
                                        kernel_size=9,
                                        depth=3,
                                        repeats=2)
        decoder = ctn.ConvTasNetDecoder(encoder_in_channel=1,
                                        decoder_out_channel=2,
                                        feature_channel=128,
                                        skipconnection_channel=16)

        target_length = 80
        input_length = encoder.reverse_length(
            decoder.reverse_length(target_length))
        output_length = decoder.forward_length(
            encoder.forward_length(input_length))

        x = torch.rand(8, 1, input_length)
        embd, X = encoder(x)
        y = decoder(embd, X)
        self.assertEqual(y.shape, (8, 2, output_length))

    def test_convtasnetdecoder_multiinput(self):
        encoder = ctn.ConvTasNetEncoder(encoder_in_channel=2,
                                        feature_channel=128,
                                        block_channel=14,
                                        bottleneck_channel=12,
                                        skipconnection_channel=16,
                                        kernel_size=9,
                                        depth=3,
                                        repeats=2)
        decoder = ctn.ConvTasNetDecoder(encoder_in_channel=2,
                                        decoder_out_channel=3,
                                        feature_channel=128,
                                        skipconnection_channel=16)

        target_length = 80
        input_length = encoder.reverse_length(
            decoder.reverse_length(target_length))
        output_length = decoder.forward_length(
            encoder.forward_length(input_length))

        x = torch.rand(8, 2, input_length)
        embd, X = encoder(x)
        y = decoder(embd, X)
        self.assertEqual(y.shape, (8, 3, 2, output_length))




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

    def test_conv_tasnet_shared_masks(self):
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
        x = torch.rand(8, 1, 16000)
        signals = ctasnet(x)
        self.assertEqual(signals.shape, (8, 2, 16000))

    def test_conv_tasnet_multi_input(self):
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
        f = ctasnet.forward_embd_feature()
        t = ctasnet.forward_embd_length(16000)
        self.assertEqual(signals.shape, (8, 3, 2, 16000))
        self.assertEqual(embd.shape, (8, f, t))

    def test_conv_tasnet_multi_input_shared_mask(self):
        # test for multi input
        ctasnet = ctn.ConvTasNet(in_channel=2,
                                 out_channel=4,
                                 feature_channel=130,
                                 block_channel=14,
                                 bottleneck_channel=12,
                                 skipconnection_channel=16,
                                 kernel_size=9,
                                 depth=3,
                                 repeats=2,
                                 mask_each=False)

        x = torch.rand(8, 2, 16000)
        signals = ctasnet(x)
        self.assertEqual(signals.shape, (8, 4, 2, 16000))

        signals, embd = ctasnet(x, return_embd=True)
        f = ctasnet.forward_embd_feature()
        t = ctasnet.forward_embd_length(16000)
        self.assertEqual(signals.shape, (8, 4, 2, 16000))
        self.assertEqual(embd.shape, (8, f, t))

    def test_conv_tasnet_codebook(self):
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

