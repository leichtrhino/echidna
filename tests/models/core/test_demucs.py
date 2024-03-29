
import unittest
import torch

from echidna.models.core import demucs as dmx
from echidna.models.multidomain.encdec import EncDecModel
from echidna.models.utils import match_length

class TestDemucsModels(unittest.TestCase):
    def test_restricted_blstm(self):
        blstm = dmx.RestrictedBiLSTM(input_size=10,
                                     output_size=30,
                                     hidden_size=20,
                                     freq_dim=1,
                                     span=11,
                                     stride=5,
                                     num_layers=2,)
        # usual case
        x = torch.rand(8, 10, 20)
        y = blstm(x)
        self.assertEqual(y.shape, (8, 30, 20))

        # input size < span
        x = torch.rand(8, 10, 9)
        y = blstm(x)
        self.assertEqual(y.shape, (8, 30, 9))

        blstm = dmx.RestrictedBiLSTM(input_size=10,
                                     output_size=30,
                                     hidden_size=20,
                                     freq_dim=13,
                                     span=11,
                                     stride=5,
                                     num_layers=2,)
        # with frequency dimension
        x = torch.rand(8, 10, 13, 9)
        y = blstm(x)
        self.assertEqual(y.shape, (8, 30, 13, 9))

    def test_local_attention(self):
        attn = dmx.LocalAttention(embd_dim=20,
                                  freq_dim=1,
                                  num_heads=4,
                                  num_penalize=5)
        x = torch.rand(8, 20, 100)
        y = attn(x)
        self.assertEqual(y.shape, (8, 20, 100))

        attn = dmx.LocalAttention(embd_dim=20,
                                  freq_dim=16,
                                  num_heads=4,
                                  num_penalize=5)
        x = torch.rand(8, 20, 16, 100)
        y = attn(x)
        self.assertEqual(y.shape, (8, 20, 16, 100))

    def test_bottleneck_block(self):
        # 3d, without attention
        block = dmx.CompressedResidualBlock(in_channel=11,
                                            out_channel=12,
                                            freq_dim=1,
                                            bottleneck_channel=6,
                                            kernel_size=3,
                                            dilation=2,)
        x = torch.rand(8, 11, 100)
        y = block(x)
        self.assertEqual(y.shape, (8, 12, 100))

        # 4d, without attention
        block = dmx.CompressedResidualBlock(in_channel=11,
                                            out_channel=12,
                                            freq_dim=100,
                                            bottleneck_channel=6,
                                            kernel_size=(3, 1),
                                            dilation=(2, 1),)
        x = torch.rand(8, 11, 100, 120)
        y = block(x)
        self.assertEqual(y.shape, (8, 12, 100, 120))

        # 3d, with attention
        block = dmx.CompressedResidualBlock(in_channel=11,
                                            out_channel=12,
                                            freq_dim=1,
                                            bottleneck_channel=6,
                                            kernel_size=3,
                                            dilation=2,
                                            lstm_layers=2,
                                            lstm_span=200,
                                            lstm_stride=100,
                                            local_attention_heads=2,
                                            local_attention_penalize=4)
        x = torch.rand(8, 11, 100)
        y = block(x)
        self.assertEqual(y.shape, (8, 12, 100))

        # 4d, with attention
        block = dmx.CompressedResidualBlock(in_channel=11,
                                            out_channel=12,
                                            freq_dim=100,
                                            bottleneck_channel=6,
                                            kernel_size=(3, 1),
                                            dilation=(2, 1),
                                            lstm_layers=2,
                                            lstm_span=200,
                                            lstm_stride=100,
                                            local_attention_heads=2,
                                            local_attention_penalize=4,)
        x = torch.rand(8, 11, 100, 120)
        y = block(x)
        self.assertEqual(y.shape, (8, 12, 100, 120))

    def test_encoder(self):
        # 3d input
        encoder = dmx.TEncoderBlock(in_channel=11,
                                    out_channel=12,
                                    kernel_size=8,
                                    stride=4,
                                    norm_groups=4,
                                    compress_layers=2,
                                    compress_channel=6,
                                    compress_kernel_size=3,
                                    compress_dilation_multiply=2,
                                    compress_lstm_layers=2,
                                    compress_lstm_span=200,
                                    compress_lstm_stride=100,
                                    compress_attention_heads=2,
                                    compress_attention_penalize=4,)
        x = torch.rand(8, 11, 100)
        y = encoder(x)
        self.assertEqual(y.shape, (8, 12, 100 // 4))
        self.assertEqual(encoder.forward_length(100), y.shape[-1])
        self.assertEqual(encoder.reverse_length(y.shape[-1]), 100)

        # 4d input
        encoder = dmx.ZEncoderBlock(in_channel=11,
                                    out_channel=12,
                                    freq_dim=100,
                                    kernel_size=8,
                                    stride=4,
                                    norm_groups=4,
                                    compress_layers=2,
                                    compress_channel=6,
                                    compress_kernel_size=3,
                                    compress_dilation_multiply=2,
                                    compress_lstm_layers=2,
                                    compress_lstm_span=200,
                                    compress_lstm_stride=100,
                                    compress_attention_heads=2,
                                    compress_attention_penalize=4,)

        x = torch.rand(8, 11, 100, 120)
        y = encoder(x)
        self.assertEqual(y.shape, (8, 12, 100 // 4, 120))
        self.assertEqual(encoder.forward_length(120), y.shape[-1])
        self.assertEqual(encoder.reverse_length(y.shape[-1]), 120)

        # set stride=kernel_size to no padding
        encoder = dmx.ZEncoderBlock(in_channel=11,
                                    out_channel=12,
                                    freq_dim=8,
                                    kernel_size=8,
                                    stride=8,
                                    norm_groups=4,
                                    compress_layers=2,
                                    compress_channel=6,
                                    compress_kernel_size=3,
                                    compress_dilation_multiply=2,
                                    compress_lstm_layers=2,
                                    compress_lstm_span=200,
                                    compress_lstm_stride=100,
                                    compress_attention_heads=2,
                                    compress_attention_penalize=4,)

        x = torch.rand(8, 11, 8, 120)
        y = encoder(x)
        self.assertEqual(y.shape, (8, 12, 8 // 8, 120))
        self.assertEqual(encoder.forward_length(120), y.shape[-1])
        self.assertEqual(encoder.reverse_length(y.shape[-1]), 120)


    def test_decoder(self):
        # 3d input
        encoder = dmx.TDecoderBlock(in_channel=11,
                                    out_channel=12,
                                    kernel_size=8,
                                    stride=4,
                                    norm_groups=4)
        x = torch.rand(8, 11, 100)
        y = encoder(x)
        self.assertEqual(y.shape, (8, 12, 100 * 4))
        self.assertEqual(encoder.forward_length(100), y.shape[-1])
        self.assertEqual(encoder.reverse_length(y.shape[-1]), 100)

        # 4d input
        encoder = dmx.ZDecoderBlock(in_channel=11,
                                    out_channel=12,
                                    freq_dim=100,
                                    kernel_size=8,
                                    stride=4,
                                    norm_groups=4)
        x = torch.rand(8, 11, 100, 120)
        y = encoder(x)
        self.assertEqual(y.shape, (8, 12, 100 * 4, 120))
        self.assertEqual(encoder.forward_length(120), y.shape[-1])
        self.assertEqual(encoder.reverse_length(y.shape[-1]), 120)


    def test_demucs_encoder(self):
        demucs = dmx.DemucsEncoder(
            # architecture parameters
            in_channel=2,
            out_channel=3,
            mid_channels=[48, 96, 144, 192],
            # conv parameters
            kernel_size=8,
            stride=4,
            inner_kernel_size=4,
            inner_stride=2,
            # misc. architecture parameters
            embedding_layers=1,
            attention_layers=2,
        )

        x = torch.rand(8, 2, 16000)
        embd, t_encs, z_encs, x_ = demucs(x)
        self.assertEqual(demucs.forward_feature_size(), 192)
        self.assertEqual(
            embd.shape,
            (
                8,
                demucs.forward_feature_size(),
                demucs.forward_length(16000)
            )
        )
        self.assertTrue(torch.all(x == x_))

        self.assertEqual(len(t_encs), len([48, 96, 144]))
        for t_enc, c in zip(t_encs, [48, 96, 144]):
            self.assertEqual(t_enc.shape[1], c)

        self.assertEqual(len(z_encs), len([48, 96, 144]))
        for z_enc, c in zip(z_encs, [48, 96, 144]):
            self.assertEqual(z_enc.shape[1], c)

        l_base = demucs.reverse_length(200)
        self.assertEqual(
            l_base,
            demucs.reverse_length(demucs.forward_length(l_base))
        )

    def test_demucs_decoder(self):
        kwargs = dict(
            # architecture parameters
            in_channel=2,
            out_channel=3,
            mid_channels=[48, 96, 144, 192],
            # conv parameters
            kernel_size=8,
            stride=4,
            inner_kernel_size=4,
            inner_stride=2,
        )

        encoder = dmx.DemucsEncoder(**kwargs)
        decoder = dmx.DemucsDecoder(**kwargs)

        x = torch.rand(8, 2, 16000)
        embd, t_encs, z_encs, x_ = encoder(x)
        y = decoder(embd, t_encs, z_encs, x_)

        self.assertEqual(
            y.shape[-1],
            decoder.forward_length(embd.shape[-1])
        )

        self.assertEqual(
            y.shape,
            (8, 3, 2, decoder.forward_length(embd.shape[-1]))
        )

        self.assertEqual(
            decoder.reverse_length(y.shape[-1]),
            embd.shape[-1]
        )

    def test_demucs_decoder_residual(self):
        encoder = dmx.DemucsEncoder(
            # architecture parameters
            in_channel=2,
            out_channel=3,
            mid_channels=[48, 96, 144, 192],
            # conv parameters
            kernel_size=8,
            stride=4,
            inner_kernel_size=4,
            inner_stride=2,
        )
        decoder = dmx.DemucsDecoder(
            # architecture parameters
            in_channel=2,
            out_channel=3,
            mid_channels=[48, 96, 144, 192],
            # conv parameters
            kernel_size=8,
            stride=4,
            inner_kernel_size=4,
            inner_stride=2,
            output_residual=True
        )

        x = torch.rand(8, 2, 16000)
        embd, t_encs, z_encs, x_ = encoder(x)
        y = decoder(embd, t_encs, z_encs, x_)

        self.assertEqual(
            y.shape[-1],
            decoder.forward_length(embd.shape[-1])
        )

        self.assertEqual(
            y.shape,
            (8, 3, 2, decoder.forward_length(embd.shape[-1]))
        )

        self.assertEqual(
            decoder.reverse_length(y.shape[-1]),
            embd.shape[-1]
        )

        self.assertLess(
            torch.max(torch.abs(
                match_length(x, y.shape[-1]) - torch.sum(y, dim=1)
            )),
            1e-3
        )

    def test_demucs(self):
        demucs = EncDecModel(
            encoder_class=dmx.DemucsEncoder,
            decoder_class=dmx.DemucsDecoder,
            hyperparameters={
                'base': dict(
                    # architecture parameters
                    in_channel=2,
                    out_channel=3,
                    mid_channels=[48, 96, 144, 192],
                    # conv parameters
                    kernel_size=8,
                    stride=4,
                    inner_kernel_size=4,
                    inner_stride=2,
                    # misc. architecture parameters
                    embedding_layers=1,
                    attention_layers=2,
                ),
            }
        )

        target_length = 16000
        input_length = demucs.reverse_wave_length(target_length)
        output_length = demucs.forward_wave_length(input_length)
        x = torch.rand(8, 2, input_length)
        y = demucs(x)
        self.assertEqual(y['waves'].shape, (8, 3, 2, output_length))

    def test_demucs_custom_kernel_size(self):
        demucs = EncDecModel(
            encoder_class=dmx.DemucsEncoder,
            decoder_class=dmx.DemucsDecoder,
            hyperparameters={
                'base': dict(
                    # architecture parameters
                    in_channel=2,
                    out_channel=3,
                    mid_channels=[48, 96, 144, 192],
                    # conv parameters
                    kernel_size=[8, 8, 4],
                    stride=[4, 4, 2],
                    inner_kernel_size=4,
                    inner_stride=2,
                    # misc. architecture parameters
                    embedding_layers=1,
                    attention_layers=2,
                )
            }
        )

        target_length = 16000
        input_length = demucs.reverse_wave_length(target_length)
        output_length = demucs.forward_wave_length(input_length)
        x = torch.rand(8, 2, input_length)
        y = demucs(x)
        self.assertEqual(y['waves'].shape, (8, 3, 2, output_length))
        self.assertEqual(demucs.encoder.stft.hop_length, 4*4*2)
        # the first factor is constant, second and third are from strides
        # last one is from kernel_size[-1]
        self.assertEqual(demucs.encoder.stft.n_fft, 2 * 4*4 * 4)
