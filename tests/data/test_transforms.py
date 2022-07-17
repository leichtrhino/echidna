
import unittest
import torch

from echidna.data import transforms

class TestTransform(unittest.TestCase):
    def test_multipointscale(self):
        x = torch.ones(100)
        scales = [1.0, 2.0, 3.0]
        fractions = [4.0, 1.0]
        y = transforms.MultiPointScale(scales, fractions)(x)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(y[0], 1.0)
        self.assertEqual(y[-1], 3.0)

    def test_crop(self):
        x = torch.ones(100)

        c1 = transforms.Crop(waveform_length=50,
                             offset=0)
        y = c1(x)
        self.assertEqual(y.shape, (50,))
        self.assertTrue(torch.all(y == torch.ones(50)))

        c2 = transforms.Crop(waveform_length=50,
                             offset=90)
        y = c2(x)
        self.assertEqual(y.shape, (50,))
        self.assertTrue(torch.all(y == torch.ones(50)))

        c3 = transforms.Crop(waveform_length=600,
                             offset=0)
        y = c3(x)
        self.assertEqual(y.shape, (600,))
        self.assertTrue(torch.all(y == torch.ones(600)))

        c4 = transforms.Crop(waveform_length=600,
                             offset=190)
        y = c4(x)
        self.assertEqual(y.shape, (600,))
        self.assertTrue(torch.all(y == torch.ones(600)))


