
import unittest
import torch

from echidna.models import convtasnet as ctn
from echidna.models.utils import match_length
from echidna.models.encoderdecoder import EncoderDecoderModel

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

        target_length = 1000
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

        target_length = 1000
        input_length = encoder.reverse_length(
            decoder.reverse_length(target_length))
        output_length = decoder.forward_length(
            encoder.forward_length(input_length))

        x = torch.rand(8, 2, input_length)
        embd, X = encoder(x)
        y = decoder(embd, X)
        self.assertEqual(y.shape, (8, 3, 2, output_length))

    def test_convtasnetdecoder_residual(self):
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
                                        skipconnection_channel=16,
                                        output_residual=True)

        target_length = 1000
        input_length = encoder.reverse_length(
            decoder.reverse_length(target_length))
        output_length = decoder.forward_length(
            encoder.forward_length(input_length))

        x = torch.rand(8, input_length)
        y = decoder(*encoder(x))

        self.assertEqual(y.shape, (8, 2, output_length))
        # check residual output
        length = min(x.shape[-1], y.shape[-1])
        self.assertLess(
            torch.max(torch.abs(
                match_length(x, length)
                - match_length(torch.sum(y, dim=-2), length)
            )).detach().item(),
            1
        )

    def test_convtasnet(self):
        convtasnet = EncoderDecoderModel(
            encoder_class=ctn.ConvTasNetEncoder,
            decoder_class=ctn.ConvTasNetDecoder,
            base_hyperparameters=dict(
                encoder_in_channel=1,
                decoder_out_channel=2,
                feature_channel=128,
                block_channel=14,
                bottleneck_channel=12,
                skipconnection_channel=16,
                kernel_size=9,
                depth=3,
                repeats=2
            ),
        )

        target_length = 1000
        input_length = convtasnet.reverse_wave_length(target_length)
        output_length = convtasnet.forward_wave_length(input_length)

        x = torch.rand(8, 1, input_length)
        y = convtasnet(x)
        self.assertEqual(y.shape, (8, 2, output_length))

