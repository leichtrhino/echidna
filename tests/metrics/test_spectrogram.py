
import unittest
import torch

from echidna.metrics.spectrogram import (
    multiscale_spectrogram_loss,
    L1SpectrogramLoss,
    L2SpectrogramLoss,
    SpectrogramConvergenceLoss,
    SpectrogramLoss,
)

class TestFunctional(unittest.TestCase):
    def test_numerical_ok(self):
        a = torch.rand((3, 2, 4000))
        b = torch.rand((3, 2, 4000))

        loss_aa = multiscale_spectrogram_loss(a, a)
        loss_ab = multiscale_spectrogram_loss(a, b)

        self.assertEqual(loss_aa.shape, (3,))
        self.assertTrue(torch.all(loss_aa < loss_ab))

    def test_fftspec_ng(self):
        a = torch.rand((3, 2, 4000))

        is_error = False
        try:
            multiscale_spectrogram_loss(a, a, fft_spec=[])
        except:
            is_error = True
        self.assertTrue(is_error)


class TestL1SpectrogramLoss(unittest.TestCase):
    def setUp(self):
        self.loss_class = L1SpectrogramLoss

    def test_numerical_ok(self):
        a = torch.rand((3, 2, 4000))
        b = torch.rand((3, 2, 4000))

        loss_aa = self.loss_class(reduction='none')(a, a)
        loss_ab = self.loss_class(reduction='none')(a, b)

        self.assertEqual(loss_aa.shape, (3,))
        self.assertTrue(torch.all(loss_aa < loss_ab))

    def test_domain_ok(self):
        loss = self.loss_class()
        self.assertEqual(loss.domains, ('waves',))

    def test_serialize_ok(self):
        loss = self.loss_class()

        ld = loss.to_dict()
        ld_type, ld = ld['type'], ld['args']
        self.assertIn('fft_spec', ld.keys())
        self.assertEqual(ld['fft_spec'], loss.fft_spec)

        loss_d = self.loss_class.from_dict({'type': ld_type, 'args': ld})
        self.assertEqual(loss.reduction, loss_d.reduction)
        self.assertEqual(loss.fft_spec, loss_d.fft_spec)

    def test_fftspec_ng(self):
        is_error = False
        try:
            self.loss_class(reduction='none', fft_spec=None)
        except:
            is_error = True
        self.assertTrue(is_error)


class TestL2SpectrogramLoss(TestL1SpectrogramLoss):
    def setUp(self):
        self.loss_class = L2SpectrogramLoss

class TestSpectrogramConvergenceLoss(TestL2SpectrogramLoss):
    def setUp(self):
        self.loss_class = SpectrogramConvergenceLoss

class TestSpectrogramConvergenceLoss(TestL2SpectrogramLoss):
    def setUp(self):
        self.loss_class = SpectrogramConvergenceLoss

class TestSpectrogramLoss(TestL2SpectrogramLoss):
    def setUp(self):
        self.loss_class = SpectrogramLoss

    def test_serialize_ok(self):
        loss = self.loss_class()

        ld = loss.to_dict()
        ld_type, ld = ld['type'], ld['args']
        self.assertEqual(ld['reduction'], loss.reduction)
        self.assertEqual(ld['fft_spec'], loss.fft_spec)
        self.assertEqual(ld['spectral_convergence_weight'],
                         loss.spectral_convergence_weight)
        self.assertEqual(ld['spectral_magnitude_weight'],
                         loss.spectral_magnitude_weight)
        self.assertEqual(ld['spectral_magnitude_norm'],
                         loss.spectral_magnitude_norm)
        self.assertEqual(ld['spectral_magnitude_log'],
                         loss.spectral_magnitude_log)

        loss_d = self.loss_class.from_dict({'type': ld_type, 'args': ld})
        self.assertEqual(loss_d.reduction, loss.reduction)
        self.assertEqual(loss_d.fft_spec, loss.fft_spec)
        self.assertEqual(loss_d.spectral_convergence_weight,
                         loss.spectral_convergence_weight)
        self.assertEqual(loss_d.spectral_magnitude_weight,
                         loss.spectral_magnitude_weight)
        self.assertEqual(loss_d.spectral_magnitude_norm,
                         loss.spectral_magnitude_norm)
        self.assertEqual(loss_d.spectral_magnitude_log,
                         loss.spectral_magnitude_log)

