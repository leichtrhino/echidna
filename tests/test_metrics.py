import unittest
import torch

import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from chimerau import metrics

class TestMetrics(unittest.TestCase):
    def test_approximation_loss(self):
        a = torch.rand((8, 2, 16000))
        b = torch.rand((8, 2, 16000))

        A = torch.stft(a.flatten(0, 1), n_fft=2048, return_complex=True)
        B = torch.stft(b.flatten(0, 1), n_fft=2048, return_complex=True)

        self.assertLess(metrics.approximation_loss(a, a),
                        metrics.approximation_loss(a, b))
        self.assertLess(metrics.approximation_loss(a, a, norm=2),
                        metrics.approximation_loss(a, b, norm=2))

        self.assertLess(metrics.approximation_loss(A, A),
                        metrics.approximation_loss(A, B))


    def test_source_to_distortion_ratio(self):
        a = torch.rand((8, 2, 16000))
        b = torch.rand((8, 2, 16000))

        # test sdr
        self.assertGreater(
            metrics.source_to_distortion_ratio(a, a),
            metrics.source_to_distortion_ratio(a, b))

        # test si-sdr
        self.assertGreater(
            metrics.source_to_distortion_ratio(a, a, scale_invariant=True),
            metrics.source_to_distortion_ratio(a, b, scale_invariant=True))
        self.assertEqual(
            metrics.source_to_distortion_ratio(a, a, scale_invariant=True),
            metrics.source_to_distortion_ratio(a, 2*a, scale_invariant=True))

        # test sd-sdr
        self.assertGreater(
            metrics.source_to_distortion_ratio(a, a, scale_dependent=True),
            metrics.source_to_distortion_ratio(a, b, scale_dependent=True))
        self.assertLessEqual(
            metrics.source_to_distortion_ratio(a, b, scale_dependent=True),
            metrics.source_to_distortion_ratio(a, b, scale_dependent=False))

    def test_multiscale_spectrogram(self):
        a = torch.rand((8, 2, 16000))
        b = torch.rand((8, 2, 16000))

        self.assertLess(
            metrics.multiscale_spectrogram_loss(a, a),
            metrics.multiscale_spectrogram_loss(a, b))

    def test_deep_clustering_loss(self):
        u = torch.rand((8, 256 * 120, 2))
        v = torch.rand((8, 256 * 120, 20))
        w = torch.rand((8, 256 * 120))

        self.assertLess(
            0, metrics.deep_clustering_loss(v, u, mode='deep-clustering'))
        self.assertLess(
            0, metrics.deep_clustering_loss(v, u, mode='deep-lda'))
        self.assertLess(
            0, metrics.deep_clustering_loss(v, u, mode='graph-laplacian-distance'))
        self.assertLess(
            0, metrics.deep_clustering_loss(v, u, mode='whitened-kmeans'))

        self.assertLess(
            0, metrics.deep_clustering_loss(v, u, w, mode='deep-clustering'))
        self.assertLess(
            0, metrics.deep_clustering_loss(v, u, w, mode='deep-lda'))
        self.assertLess(
            0, metrics.deep_clustering_loss(v, u, w, mode='graph-laplacian-distance'))
        self.assertLess(
            0, metrics.deep_clustering_loss(v, u, w, mode='whitened-kmeans'))

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
        m = metrics.permutation_invariant(metrics.approximation_loss)

        self.assertEqual(m(a, a), m(a, a_t))
        self.assertLess(m(a, a_t), m(a, b))
        self.assertEqual(m(A, A), m(A, A_t))
        self.assertLess(m(A, A_t), m(A, B))

        # sdr
        m = metrics.permutation_invariant(metrics.source_to_distortion_ratio,
                                          aggregate_perm_fn=max,
                                          aggregate_loss_fn=sum)

        self.assertEqual(m(a, a), m(a, a_t))
        self.assertGreater(m(a, a_t), m(a, b))

if __name__ == '__main__':
    unittest.main()
