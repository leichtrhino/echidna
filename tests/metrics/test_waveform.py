
import unittest
import torch

from echidna.metrics.waveform import (
    source_to_distortion_ratio,
    NegativeSDRLoss,
    NegativeSISDRLoss,
    NegativeSDSDRLoss,
    L1WaveformLoss,
    L2WaveformLoss,
)

class TestFunctional(unittest.TestCase):
    def test_sdr_ok(self):
        a = torch.rand(3, 2, 4000)
        b = torch.rand(3, 2, 4000)

        sdr_aa = source_to_distortion_ratio(a, a)
        sdr_ab = source_to_distortion_ratio(a, b)

        # test sdr
        self.assertEqual(sdr_aa.shape, (3,))
        self.assertTrue(torch.all(sdr_aa > sdr_ab))

    def test_sisdr_ok(self):
        a = torch.rand(3, 2, 4000)
        b = torch.rand(3, 2, 4000)

        sdr_aa = source_to_distortion_ratio(a, a, scale_invariant=True)
        sdr_a2a = source_to_distortion_ratio(a, 2*a, scale_invariant=True)
        sdr_ab = source_to_distortion_ratio(a, b, scale_invariant=True)

        # test si-sdr
        self.assertEqual(sdr_aa.shape, (3,))
        self.assertTrue(torch.all(sdr_aa > sdr_ab))
        self.assertTrue(torch.all(sdr_aa == sdr_a2a))

    def test_sdsdr_ok(self):
        a = torch.rand(3, 2, 4000)
        b = torch.rand(3, 2, 4000)

        sdr_aa = source_to_distortion_ratio(a, a, scale_dependent=True)
        sdr_ab = source_to_distortion_ratio(a, b, scale_dependent=True)
        snr_ab = source_to_distortion_ratio(a, b, scale_invariant=False)

        # test sd-sdr
        self.assertEqual(sdr_aa.shape, (3,))
        self.assertTrue(torch.all(sdr_aa > sdr_ab))
        self.assertTrue(torch.all(sdr_ab <= snr_ab))

    def test_alg_ng(self):
        a = torch.rand(3, 2, 4000)
        b = torch.rand(3, 2, 4000)

        is_error = False
        try:
            source_to_distortion_ratio(a,
                                       b,
                                       scale_invariant=True,
                                       scale_dependent=True,)
        except:
            is_error = True
        self.assertTrue(is_error)


class TestNegativeSDRLoss(unittest.TestCase):
    def test_numerical_ok(self):
        a = torch.rand(3, 2, 4000)
        b = torch.rand(3, 2, 4000)

        sdr_aa = NegativeSDRLoss('none')(a, a)
        sdr_ab = NegativeSDRLoss('none')(a, b)
        self.assertEqual(sdr_aa.shape, (3,))
        self.assertTrue(torch.all(sdr_aa < sdr_ab))

    def test_domain(self):
        self.assertEqual(NegativeSDRLoss('none').domains, ('waves',))


class TestNegativeSISDRLoss(unittest.TestCase):
    def test_numerical_ok(self):
        a = torch.rand(3, 2, 4000)
        b = torch.rand(3, 2, 4000)

        sdr_aa = NegativeSISDRLoss('none')(a, a)
        sdr_a2a = NegativeSISDRLoss('none')(a, 2*a)
        sdr_ab = NegativeSISDRLoss('none')(a, b)

        # test si-sdr
        self.assertEqual(sdr_aa.shape, (3,))
        self.assertTrue(torch.all(sdr_aa < sdr_ab))
        self.assertTrue(torch.all(sdr_aa == sdr_a2a))

    def test_domain(self):
        self.assertEqual(NegativeSISDRLoss('none').domains, ('waves',))


class TestNegativeSDSDRLoss(unittest.TestCase):
    def test_numerical_ok(self):
        a = torch.rand(3, 2, 4000)
        b = torch.rand(3, 2, 4000)

        sdr_aa = NegativeSDSDRLoss('none')(a, a)
        sdr_ab = NegativeSDSDRLoss('none')(a, b)
        snr_ab = NegativeSDRLoss('none')(a, b)

        # test sd-sdr
        self.assertEqual(sdr_aa.shape, (3,))
        self.assertTrue(torch.all(sdr_aa < sdr_ab))
        self.assertTrue(torch.all(sdr_ab >= snr_ab))

    def test_domain(self):
        self.assertEqual(NegativeSDSDRLoss('none').domains, ('waves',))

class TestL1WaveformLoss(unittest.TestCase):
    def setUp(self):
        self.loss_class = L1WaveformLoss

    def test_numerical_ok(self):
        a = torch.rand(3, 2, 4000)
        b = torch.rand(3, 2, 4000)

        loss_aa = self.loss_class('none')(a, a)
        loss_ab = self.loss_class('none')(a, b)

        self.assertEqual(loss_aa.shape, (3,))
        self.assertTrue(torch.all(loss_aa < loss_ab))

    def test_domain(self):
        self.assertEqual(NegativeSDSDRLoss('none').domains, ('waves',))

class TestL2WaveformLoss(TestL1WaveformLoss):
    def setUp(self):
        self.loss_class = L2WaveformLoss
