import unittest
import torch

from echidna import metrics as mt

class TestMt(unittest.TestCase):
    def test_source_to_distortion_ratio(self):
        a = torch.rand((8, 2, 16000))
        b = torch.rand((8, 2, 16000))

        # test sdr
        self.assertGreater(
            mt.source_to_distortion_ratio(a, a),
            mt.source_to_distortion_ratio(a, b))

        # test si-sdr
        self.assertGreater(
            mt.source_to_distortion_ratio(a, a, scale_invariant=True),
            mt.source_to_distortion_ratio(a, b, scale_invariant=True))
        self.assertEqual(
            mt.source_to_distortion_ratio(a, a, scale_invariant=True),
            mt.source_to_distortion_ratio(a, 2*a, scale_invariant=True))

        # test sd-sdr
        self.assertGreater(
            mt.source_to_distortion_ratio(a, a, scale_dependent=True),
            mt.source_to_distortion_ratio(a, b, scale_dependent=True))
        self.assertLessEqual(
            mt.source_to_distortion_ratio(a, b, scale_dependent=True),
            mt.source_to_distortion_ratio(a, b, scale_dependent=False))

    def test_multiscale_spectrogram(self):
        a = torch.rand((8, 2, 16000))
        b = torch.rand((8, 2, 16000))

        self.assertLess(
            mt.multiscale_spectrogram_loss(a, a),
            mt.multiscale_spectrogram_loss(a, b))

    def test_deep_clustering_loss(self):
        u = torch.rand((8, 256 * 120, 2))
        v = torch.rand((8, 256 * 120, 20))
        w = torch.rand((8, 256 * 120))

        self.assertLess(
            0, mt.deep_clustering_loss(v, u, mode='deep-clustering'))
        self.assertLess(
            0, mt.deep_clustering_loss(v, u, mode='deep-lda'))
        self.assertLess(
            0, mt.deep_clustering_loss(v, u, mode='graph-laplacian-distance'))
        self.assertLess(
            0, mt.deep_clustering_loss(v, u, mode='whitened-kmeans'))

        self.assertLess(
            0, mt.deep_clustering_loss(v, u, w, mode='deep-clustering'))
        self.assertLess(
            0, mt.deep_clustering_loss(v, u, w, mode='deep-lda'))
        self.assertLess(
            0, mt.deep_clustering_loss(v, u, w, mode='graph-laplacian-distance'))
        self.assertLess(
            0, mt.deep_clustering_loss(v, u, w, mode='whitened-kmeans'))

    def test_permutation_invariant(self):
        a = torch.rand((8, 2, 16000))
        b = torch.rand((8, 2, 16000))
        a_t = torch.stack((a[:, 1, :], a[:, 0, :]), dim=1)

        A = torch.stft(a.flatten(0, 1), n_fft=2048, return_complex=True)\
                 .unflatten(0, (8, 2))
        B = torch.stft(b.flatten(0, 1), n_fft=2048, return_complex=True)\
                 .unflatten(0, (8, 2))
        A_t = torch.stft(a_t.flatten(0, 1), n_fft=2048, return_complex=True)\
                   .unflatten(0, (8, 2))

        # loss
        m = mt.permutation_invariant(lambda x, y: torch.abs(x - y).sum())

        self.assertEqual(m(a, a), m(a, a_t))
        self.assertLess(m(a, a_t), m(a, b))
        self.assertEqual(m(A, A), m(A, A_t))
        self.assertLess(m(A, A_t), m(A, B))

        # sdr
        m = mt.permutation_invariant(mt.source_to_distortion_ratio,
                                     aggregate_perm_fn=max,
                                     aggregate_loss_fn=sum)

        self.assertEqual(m(a, a), m(a, a_t))
        self.assertGreater(m(a, a_t), m(a, b))

