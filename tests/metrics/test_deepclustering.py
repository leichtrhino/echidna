
import unittest
import torch

from echidna.metrics.deepclustering import (
    deep_clustering_loss,
    DeepClusteringLoss,
    DeepLDALoss,
    GraphLaplacianLoss,
    WhitenedKMeansLoss,
)

class TestFunctional(unittest.TestCase):
    def validate_mode(self, mode):
        u = torch.rand((3, 16 * 32, 2))
        v = torch.rand((3, 16 * 32, 20))
        w = torch.rand((3, 16 * 32))

        self.assertEqual(deep_clustering_loss(v, u, mode=mode).shape, (3,))
        self.assertTrue(torch.all(0<deep_clustering_loss(v, u, mode=mode)))
        self.assertTrue(torch.all(0<deep_clustering_loss(v, u, w, mode=mode)))

    def test_deep_clustering_ok(self):
        self.validate_mode('deep_clustering')

    def test_deep_lda_ok(self):
        self.validate_mode('deep_lda')

    def test_graph_laplacian_distance_ok(self):
        self.validate_mode('graph_laplacian_distance')

    def test_whitened_kmeans_ok(self):
        self.validate_mode('whitened_kmeans')

    def test_invalid_mode_ng(self):
        u = torch.rand((3, 16 * 32, 2))
        v = torch.rand((3, 16 * 32, 20))
        w = torch.rand((3, 16 * 32))
        mode = 'invalid'

        is_error = False
        try:
            deep_clustering_loss(v, u, mode=mode)
        except:
            is_error = True
        self.assertTrue(is_error)


class TestDeepClusteringLoss(unittest.TestCase):
    def setUp(self):
        self.dc_class = DeepClusteringLoss

    def test_numerical_ok(self):
        u = torch.rand((3, 16 * 32, 2))
        v = torch.rand((3, 16 * 32, 20))
        w = torch.rand((3, 16 * 32))
        mode = 'deep_clustering'

        dc_loss = self.dc_class(label='argmax',
                                weight='none',
                                reduction='none')

        self.assertEqual(dc_loss(v, u).shape, (3,))
        self.assertTrue(torch.all(0 < dc_loss(v, u)))

        dc_loss = self.dc_class(label='softmax',
                                weight='magnitude_ratio',
                                reduction='none')

        self.assertEqual(dc_loss(v, u).shape, (3,))
        self.assertTrue(torch.all(0 < dc_loss(v, u)))

    def test_domain_ok(self):
        self.assertEqual(self.dc_class().domains, ('embd',))

    def test_serialize_ok(self):
        loss = self.dc_class()

        ld = loss.to_dict()
        ld_type, ld = ld['type'], ld['args']
        self.assertIn('label', ld.keys())
        self.assertEqual(ld['label'], loss.label)
        self.assertIn('weight', ld.keys())
        self.assertEqual(ld['weight'], loss.weight)

        loss_d = self.dc_class.from_dict({'type': ld_type, 'args': ld})
        self.assertEqual(loss.reduction, loss_d.reduction)
        self.assertEqual(loss.label, loss_d.label)
        self.assertEqual(loss.weight, loss_d.weight)

    def test_param_ng(self):
        is_error = False
        try:
            dc_loss = self.dc_class(label='invalid',
                                    weight='magnitude_ratio',
                                    reduction='none')
        except:
            is_error = True
        self.assertTrue(is_error)

        is_error = False
        try:
            dc_loss = self.dc_class(label='softmax',
                                    weight='invalid',
                                    reduction='none')
        except:
            is_error = True
        self.assertTrue(is_error)

class TestDeepLDALoss(TestDeepClusteringLoss):
    def setup(self):
        self.dc_class = DeepLDALoss

class TestGraphLaplacianLoss(TestDeepClusteringLoss):
    def setup(self):
        self.dc_class = GraphLaplacianLoss

class TestWhitenedKMeansLoss(TestDeepClusteringLoss):
    def setUp(self):
        self.dc_class = WhitenedKMeansLoss

