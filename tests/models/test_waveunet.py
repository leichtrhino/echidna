
import unittest
import math
import itertools
import torch

from echidna.models import waveunet as wu
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
        d = wu.WaveUNetDecoder(upsampling_channel_in=96,
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

        self.assertEqual(y.shape, torch.Size((8, 3, output_length)))

    def test_waveunet_(self):
        m = EncoderDecoderModel(
            encoder_class=wu.WaveUNetEncoder,
            encoder_params=dict(
                channel_in=2,
                downsampling_channel_out=[24, 48, 72],
                downsampling_kernel_size=15,
                downsampling_rate=2,
                encoder_channel_out=[96],
                encoder_kernel_size=15
            ),
            decoder_class=wu.WaveUNetDecoder,
            decoder_params=dict(
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
        self.assertEqual(m(x).shape, (8, 3, output_length))


    def test_encoder(self):
        e = wu.Encoder(channel_in=2,
                       channel_out=[24, 48, 72],
                       kernel_size=15,
                       downsample_rate=2)
        l_in = e.reverse_length(125)
        x = torch.zeros((8, 2, l_in))
        y = e(x)
        self.assertEqual(y.shape, torch.Size((8, 72, 125)))
        self.assertEqual(e.forward_length(l_in), y.shape[-1])

        ys = e(x, return_each=True)
        for c, y in zip((24, 48, 72), ys):
            self.assertEqual(y.shape[:-1], torch.Size((8, c)))

    def test_decoder(self):
        poc = wu._pad_or_crop
        x = torch.arange(1, 4).reshape(1, 1, 3)
        self.assertEqual(poc(x, 1).tolist(), [[[      2      ]]])
        self.assertEqual(poc(x, 2).tolist(), [[[   1, 2      ]]])
        self.assertEqual(poc(x, 3).tolist(), [[[   1, 2, 3   ]]])
        self.assertEqual(poc(x, 4).tolist(), [[[1, 1, 2, 3   ]]])
        self.assertEqual(poc(x, 5).tolist(), [[[1, 1, 2, 3, 3]]])

        d = wu.Decoder(channel_in=72,
                       channel_out=[48, 24, 1],
                       kernel_size=5,
                       upsample_rate=2)
        l_in = d.reverse_length(1000)
        x = torch.zeros(8, 72, l_in)
        y = d(x)
        self.assertGreaterEqual(y.shape[-1], 1000)
        self.assertEqual(y.shape[:-1], torch.Size((8, 1)))
        l_out = y.shape[-1]

        d = wu.Decoder(channel_in=72,
                       channel_out=[48, 24, 1],
                       kernel_size=5,
                       upsample_rate=2,
                       residual_mode='concat',
                       residual_channel=[24, 12])
        self.assertEqual(
            d.layers[1].conv.weight.shape,
            torch.Size((24, 48+24, 5))
        )

        r = [torch.zeros(8, c, 10000) for c in [24, 12]]
        y = d(x, r)
        self.assertEqual(y.shape, torch.Size((8, 1, l_out)))
        r = [torch.zeros(8, c, 1) for c in [24, 12]]
        y = d(x, r)
        self.assertEqual(y.shape, torch.Size((8, 1, l_out)))

        d = wu.Decoder(channel_in=72,
                       channel_out=[48, 24, 1],
                       kernel_size=5,
                       upsample_rate=2,
                       residual_mode='add')
        self.assertEqual(
            d.layers[0].conv.weight.shape,
            torch.Size((48, 72, 5))
        )

        r = [torch.zeros(8, c, 10000) for c in [48, 24]]
        y = d(x, r)
        self.assertEqual(y.shape, torch.Size((8, 1, l_out)))
        r = [torch.zeros(8, c, 1) for c in [48, 24]]
        y = d(x, r)
        self.assertEqual(y.shape, torch.Size((8, 1, l_out)))

    def test_waveunet(self):
        n = wu.WaveUNet(
            channel_in=1,
            channel_out=2,
            channel_enc_dec=[24, 48, 72, 96],
            channel_mid=[],
            channel_out_m1=24,
            kernel_size_d=15,
            kernel_size_u=5,
            kernel_size_m=1,
            kernel_size_o=15,
            downsample_rate=2,
            upsample_mode='nearest',
            residual_mode='add',
        )

        x = torch.zeros(8, 1, 1000)
        y, e = n(x, return_embd=True)
        l_out = y.shape[-1]
        self.assertEqual(y.shape, torch.Size((8, 2, l_out)))
        self.assertEqual(e.shape[:-1], torch.Size((8, 96)))
        self.assertEqual(n.forward_length(1000), l_out)
        self.assertEqual(n.forward_embd_feature(), e.shape[1])
        self.assertEqual(n.forward_embd_length(1000), e.shape[2])

        n = wu.WaveUNet(
            channel_in=1,
            channel_out=2,
            channel_enc_dec=[24, 48, 72, 96],
            channel_mid=[96, 96],
            channel_out_m1=24,
            kernel_size_d=15,
            kernel_size_u=5,
            kernel_size_m=15,
            kernel_size_o=27,
            downsample_rate=2,
            upsample_mode='trainable',
            residual_mode='concat',
        )

        x = torch.zeros(8, 1, 1000)
        y, e = n(x, return_embd=True)
        l_out = y.shape[-1]
        self.assertEqual(y.shape, torch.Size((8, 2, l_out)))
        self.assertEqual(e.shape[:-1], torch.Size((8, 96)))

    def test_chimera_waveunet(self):
        n = wu.ChimeraWaveUNet(
            channel_in=1,
            channel_out=2,
            channel_enc_dec=[24, 48, 72, 96],
            channel_mid=[96, 96],
            channel_out_m1=24,
            embd_feature=129,
            embd_dim=20,
            kernel_size_d=15,
            kernel_size_u=5,
            kernel_size_m=15,
            kernel_size_o=27,
            downsample_rate=2,
            upsample_mode='nearest',
            residual_mode='add',
        )

        x = torch.zeros(8, 1, 1000)
        y, e = n(x)
        l_out = y.shape[-1]
        self.assertEqual(y.shape, torch.Size((8, 2, l_out)))
        self.assertEqual(e.shape, torch.Size((8, 129, e.shape[2], 20)))
        self.assertEqual(n.forward_embd_feature(), e.shape[1])
        self.assertEqual(n.forward_embd_length(1000), e.shape[2])

