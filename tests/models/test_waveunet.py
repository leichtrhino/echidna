
import unittest
import math
import itertools
import torch

from echidna.models import waveunet as wu
from echidna.models.utils import match_length
from echidna.models.encoderdecoder import EncoderDecoderModel

class TestWaveUNetModels(unittest.TestCase):
    def test_interpolate_shape(self):
        # check shape validity
        self.assertEqual(
            wu.Interpolation(2).forward_length(201), (201-1)*2+1)
        self.assertEqual(
            wu.Interpolation(4).forward_length(201), (201-1)*4+1)
        self.assertEqual(
            wu.Interpolation(2).reverse_length((201-1)*2+1), 201)
        self.assertEqual(
            wu.Interpolation(4).reverse_length((201-1)*4+1), 201)

        x = torch.zeros(3, 18, 201)
        y = wu.Interpolation(2, mode='nearest')(x)
        self.assertEqual(y.shape, torch.Size((3, 18, (201-1)*2+1)))
        y = wu.Interpolation(2, mode='linear')(x)
        self.assertEqual(y.shape, torch.Size((3, 18, (201-1)*2+1)))
        y = wu.Interpolation(2, mode='trainable', channel=18)(x)
        self.assertEqual(y.shape, torch.Size((3, 18, (201-1)*2+1)))

    def test_interpolate_numeric(self):
        # check numerical validity
        x = torch.zeros(1, 2, 2)
        x[..., 1] = 1 * 10

        i_nn = wu.Interpolation(8, mode='nearest')
        y = i_nn(x)
        self.assertTrue(torch.all(y[..., :4] == 0 * 10))
        self.assertTrue(torch.all(y[..., 5:] == 1 * 10))
        self.assertTrue(torch.all(y[..., 4] == 0.5 * 10))

        i_li = wu.Interpolation(8, mode='linear')
        y = i_li(x)
        for i in range(9):
            self.assertTrue(torch.all(y[..., i] == i / 8 * 10))

        i_tr = wu.Interpolation(8, mode='trainable', channel=2)
        y = i_tr(x)
        for i in range(9):
            self.assertTrue(torch.all(y[..., i] == i / 8 * 10))
        self.assertTrue(y.requires_grad)

    def test_interpolate_length(self):
        # check reverse length
        for r in (2, 4, 8):
            i = wu.Interpolation(r)
            for l_in in range(100, 301, 17):
                l_out = i(torch.zeros((1, 1, l_in))).shape[-1]
                l_in_pred = i.reverse_length(l_out)
                l_out_pred = i(torch.zeros(1, 1, l_in_pred)).shape[-1]
                self.assertLessEqual(l_in_pred, l_in)
                self.assertEqual(l_out_pred, l_out)
                self.assertEqual(i.forward_length(l_in), l_out)
            for l_out in range(100, 301, 17):
                l_in_pred = i.reverse_length(l_out)
                l_out_pred = i(torch.zeros(1, 1, l_in_pred)).shape[-1]
                self.assertGreaterEqual(l_out_pred, l_out)
                self.assertEqual(i.forward_length(l_in_pred), l_out_pred)

    def test_downsamplingblock_shape(self):
        # test shape validity
        x = torch.zeros(8, 13, 201)
        ds = wu.DownsamplingBlock(channel_in=13,
                                  channel_out=26,
                                  kernel_size=5,
                                  downsample_rate=4)

        self.assertEqual(ds.reverse_length(math.ceil((201-5+1)/4)), 201)
        self.assertEqual(ds.forward_length(201), math.ceil((201-5+1)/4))

        y, e = ds(x)
        self.assertEqual(y.shape, torch.Size((8, 26, math.ceil((201-5+1)/4))))
        self.assertEqual(e.shape, torch.Size((8, 26, 201-5+1)))

    def test_downsamplingblock_length(self):
        # check reverse length
        for r, k in itertools.product((2, 4, 8), (3, 5, 7)):
            b = wu.DownsamplingBlock(1, 1, k, r)
            for l_in in range(100, 301, 17):
                l_out = b(torch.zeros((1, 1, l_in)))[0].shape[-1]
                l_in_pred = b.reverse_length(l_out)
                l_out_pred = b(torch.zeros(1, 1, l_in_pred))[0].shape[-1]
                self.assertLessEqual(l_in_pred, l_in)
                self.assertEqual(l_out_pred, l_out)
                self.assertEqual(b.forward_length(l_in), l_out)
            for l_out in range(100, 301, 17):
                l_in_pred = b.reverse_length(l_out)
                l_out_pred = b(torch.zeros(1, 1, l_in_pred))[0].shape[-1]
                self.assertGreaterEqual(l_out_pred, l_out)
                self.assertEqual(b.forward_length(l_in_pred), l_out_pred)

    def test_upsamplingblock_shape(self):
        # test shape validity
        x = torch.zeros(8, 13, 201)
        e = torch.zeros(8, 13, 201*4)
        us = wu.UpsamplingBlock(channel_in=13,
                                channel_residual=13,
                                channel_out=26,
                                kernel_size=5,
                                upsample_rate=4)

        self.assertEqual(us.reverse_length((201-1)*4+1-5+1), 201)
        self.assertEqual(us.forward_length(201), (201-1)*4+1-5+1)

        y = us(x, e)
        self.assertEqual(y.shape, torch.Size((8, 26, (201-1)*4+1-5+1)))

    def test_upsamplingblock_length(self):
        # check reverse length
        for r, k in itertools.product((2, 4, 8), (3, 5, 7)):
            b = wu.UpsamplingBlock(1, 1, 1, k, r)
            for l_in in range(100, 301, 17):
                l_out = b(torch.zeros((1, 1, l_in)),
                          torch.zeros((1, 1, l_in*r))).shape[-1]
                l_in_pred = b.reverse_length(l_out)
                l_out_pred = b(torch.zeros(1, 1, l_in_pred),
                               torch.zeros(1, 1, l_in_pred*r)).shape[-1]
                self.assertLessEqual(l_in_pred, l_in)
                self.assertEqual(l_out_pred, l_out)
                self.assertEqual(b.forward_length(l_in), l_out)
            for l_out in range(100, 301, 17):
                l_in_pred = b.reverse_length(l_out)
                l_out_pred = b(torch.zeros(1, 1, l_in_pred),
                               torch.zeros(1, 1, l_in_pred*r)).shape[-1]
                self.assertGreaterEqual(l_out_pred, l_out)
                self.assertEqual(b.forward_length(l_in_pred), l_out_pred)

    def test_waveunetencoder(self):
        e = wu.WaveUNetEncoder(channel_in=2,
                               downsampling_channel_out=[24, 48, 72],
                               downsampling_kernel_size=15,
                               downsampling_rate=2,
                               encoder_channel_out=[96],
                               encoder_kernel_size=15)

        l_in = e.reverse_length(125)
        x = torch.zeros((8, 2, l_in))
        y, ys, x_ = e(x)
        self.assertEqual(x_.shape, x.shape)
        self.assertEqual(y.shape, torch.Size((8, 96, 125)))
        self.assertEqual(e.forward_length(l_in), y.shape[-1])

        for c, y in zip((24, 48, 72), ys):
            self.assertEqual(y.shape[:-1], torch.Size((8, c)))

    def test_waveunetdecoder(self):
        e = wu.WaveUNetEncoder(channel_in=2,
                               downsampling_channel_out=[24, 48, 72],
                               downsampling_kernel_size=15,
                               downsampling_rate=2,
                               encoder_channel_out=[96],
                               encoder_kernel_size=15)
        d = wu.WaveUNetDecoder(channel_in=2,
                               upsampling_channel_in=96,
                               upsampling_channel_out=[72, 48, 24],
                               upsampling_kernel_size=5,
                               upsampling_rate=2,
                               upsampling_mode='nearest',
                               upsampling_residual_channel=[72, 48, 24],
                               decoder_channel_out=3,
                               decoder_kernel_size=5,
                               decoder_residual_channel=2)

        target_length = 125
        input_length = e.reverse_length(d.reverse_length(target_length))
        output_length = d.forward_length(e.forward_length(input_length))
        x = torch.zeros((8, 2, input_length))
        y, ys, x_ = e(x)
        y = d(y, ys, x_)

        self.assertEqual(y.shape, torch.Size((8, 3, 2, output_length)))

    def test_waveunetdecoder_residual_shape(self):
        e = wu.WaveUNetEncoder(channel_in=2,
                               downsampling_channel_out=[24, 48, 72],
                               downsampling_kernel_size=15,
                               downsampling_rate=2,
                               encoder_channel_out=[96],
                               encoder_kernel_size=15)
        d = wu.WaveUNetDecoder(channel_in=2,
                               upsampling_channel_in=96,
                               upsampling_channel_out=[72, 48, 24],
                               upsampling_kernel_size=5,
                               upsampling_rate=2,
                               upsampling_mode='nearest',
                               upsampling_residual_channel=[72, 48, 24],
                               decoder_channel_out=3,
                               decoder_kernel_size=5,
                               decoder_residual_channel=2,
                               output_residual=True)

        target_length = 125
        input_length = e.reverse_length(d.reverse_length(target_length))
        output_length = d.forward_length(e.forward_length(input_length))
        x = torch.zeros((8, 2, input_length))
        y, ys, x_ = e(x)
        y = d(y, ys, x_)

        self.assertEqual(y.shape, torch.Size((8, 3, 2, output_length)))

    def test_waveunetdecoder_residual_numeric(self):
        e = wu.WaveUNetEncoder(channel_in=1,
                               downsampling_channel_out=[24, 48, 72],
                               downsampling_kernel_size=15,
                               downsampling_rate=2,
                               encoder_channel_out=[96],
                               encoder_kernel_size=15)
        d = wu.WaveUNetDecoder(channel_in=1,
                               upsampling_channel_in=96,
                               upsampling_channel_out=[72, 48, 24],
                               upsampling_kernel_size=5,
                               upsampling_rate=2,
                               upsampling_mode='nearest',
                               upsampling_residual_channel=[72, 48, 24],
                               decoder_channel_out=2,
                               decoder_kernel_size=5,
                               decoder_residual_channel=1,
                               output_residual=True)

        target_length = 125
        input_length = e.reverse_length(d.reverse_length(target_length))
        output_length = d.forward_length(e.forward_length(input_length))
        x = torch.rand((8, input_length))
        y = d(*e(x))

        self.assertEqual(y.shape, torch.Size((8, 2, output_length)))
        # check residual output
        self.assertLess(
            torch.max(torch.abs(
                match_length(x, y.shape[-1]) - torch.sum(y, dim=-2)
            )).detach().item(),
            1e-3
        )

    def test_waveunet(self):
        m = EncoderDecoderModel(
            encoder_class=wu.WaveUNetEncoder,
            decoder_class=wu.WaveUNetDecoder,
            base_hyperparameters=dict(
                channel_in=2,
                downsampling_channel_out=[24, 48, 72],
                downsampling_kernel_size=15,
                downsampling_rate=2,
                encoder_channel_out=[96],
                encoder_kernel_size=15,
                upsampling_channel_in=96,
                upsampling_channel_out=[72, 48, 24],
                upsampling_kernel_size=5,
                upsampling_rate=2,
                upsampling_mode='nearest',
                upsampling_residual_channel=[72, 48, 24],
                decoder_channel_out=3,
                decoder_kernel_size=5,
                decoder_residual_channel=2
            )
        )
        target_length = 1000
        input_length = m.reverse_wave_length(target_length)
        output_length = m.forward_wave_length(input_length)
        x = torch.zeros((8, 2, input_length))
        self.assertEqual(m(x).shape, (8, 3, 2, output_length))

