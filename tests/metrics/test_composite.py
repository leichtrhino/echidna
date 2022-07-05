
import unittest
import torch
from itertools import product

from echidna.metrics.waveform import L1WaveformLoss, L2WaveformLoss
from echidna.metrics.deepclustering import DeepClusteringLoss
from echidna.metrics.composite import CompositeLoss

class TestComposite(unittest.TestCase):
    def test_init_ok(self):
        loss_fn = CompositeLoss(
            [{'func': L1WaveformLoss(), 'weight': 1.0},
             {'func': L2WaveformLoss(), 'weight': 1.0},
             {'func': DeepClusteringLoss(), 'weight': 1.0}],
            permutation='min',
            reduction='mean',
        )

    def test_domain_ok(self):
        loss_fn = CompositeLoss(
            [{'func': L1WaveformLoss(), 'weight': 1.0},
             {'func': L2WaveformLoss(), 'weight': 1.0},
             {'func': DeepClusteringLoss(), 'weight': 1.0}],
            permutation='min',
            reduction='mean',
        )
        self.assertTrue(loss_fn.domains, set(('waves', 'embd')))

    def test_serialize_ok(self):
        loss_fn = CompositeLoss(
            [{'func': L1WaveformLoss(), 'weight': 1.0},
             {'func': L2WaveformLoss(), 'weight': 1.0},
             {'func': DeepClusteringLoss(), 'weight': 1.0}],
            permutation='min',
            reduction='mean',
        )

        # serialize
        ld = loss_fn.to_dict()
        self.assertIn('components', ld.keys())
        self.assertEqual(len(ld['components']), 3)
        for fn in ld['components']:
            self.assertIn('func', fn)
            self.assertIn('param', fn)
            self.assertIn('weight', fn)
        self.assertIn('permutation', ld.keys())
        self.assertIn('reduction', ld.keys())

        # deserialize
        loss_d = CompositeLoss.from_dict(ld)
        self.assertEqual(len(loss_fn.components), len(loss_d.components))
        for fn1, fn2 in zip(loss_fn.components, loss_d.components):
            self.assertEqual(type(fn1['func']), type(fn2['func']))
            self.assertEqual(fn1['func'].to_dict(), fn2['func'].to_dict())
            self.assertEqual(fn1['weight'], fn2['weight'])
        self.assertEqual(loss_fn.permutation, loss_d.permutation)
        self.assertEqual(loss_fn.reduction, loss_d.reduction)

    def test_forward_no_reduction_func_ok(self):
        y_pred = {
            'waves': torch.rand(3, 2, 8),
            'embd': torch.rand(3, 4, 3, 20)
        }
        y_true = {
            'waves': torch.rand(3, 2, 8),
            'embd': torch.rand(3, 4, 3, 2)
        }

        # 1. permutations = none
        loss_fn = CompositeLoss(
            [{'func': L1WaveformLoss(), 'weight': 1.0},
             {'func': L2WaveformLoss(), 'weight': 1.0},
             {'func': DeepClusteringLoss(), 'weight': 1.0}],
            permutation='none',
            reduction='mean',
        )

        loss_dict = loss_fn.forward_no_reduction(y_pred, y_true)
        self.assertEqual(type(loss_dict), dict)
        self.assertIn('batch', loss_dict)
        self.assertEqual(type(loss_dict['batch']), torch.Tensor)
        self.assertEqual(loss_dict['batch'].shape, (3,))
        self.assertIn('sample', loss_dict)
        self.assertEqual(len(loss_dict['sample']), 3)
        for d in loss_dict['sample']:
            self.assertEqual(type(d), dict)
            self.assertIn('l1_waveform', d)
            self.assertIn('l2_waveform', d)
            self.assertIn('deep_clustering', d)

        # 2. permutations = min
        loss_fn = CompositeLoss(
            [{'func': L1WaveformLoss(), 'weight': 1.0},
             {'func': L2WaveformLoss(), 'weight': 1.0},
             {'func': DeepClusteringLoss(), 'weight': 1.0}],
            permutation='min',
            reduction='mean',
        )

        loss_dict = loss_fn.forward_no_reduction(y_pred, y_true)
        self.assertEqual(type(loss_dict), dict)
        self.assertIn('batch', loss_dict)
        self.assertEqual(type(loss_dict['batch']), torch.Tensor)
        self.assertEqual(loss_dict['batch'].shape, (3,))
        self.assertIn('sample', loss_dict)
        self.assertEqual(len(loss_dict['sample']), 3)
        for d in loss_dict['sample']:
            self.assertEqual(type(d), dict)
            self.assertIn('l1_waveform', d)
            self.assertIn('l2_waveform', d)
            self.assertIn('deep_clustering', d)

        # 3. permutation = min and embedding function only
        loss_fn = CompositeLoss(
            [{'func': DeepClusteringLoss(), 'weight': 1.0}],
            permutation='min',
            reduction='mean',
        )

        loss_dict = loss_fn.forward_no_reduction(y_pred, y_true)
        self.assertEqual(type(loss_dict), dict)
        self.assertIn('batch', loss_dict)
        self.assertEqual(type(loss_dict['batch']), torch.Tensor)
        self.assertEqual(loss_dict['batch'].shape, (3,))
        self.assertIn('sample', loss_dict)
        self.assertEqual(len(loss_dict['sample']), 3)
        for d in loss_dict['sample']:
            self.assertEqual(type(d), dict)
            self.assertIn('deep_clustering', d)


    def test_forward_no_reduction_numeric_ok(self):
        s1 = torch.stack((torch.ones(3, 4), torch.zeros(3, 4)), dim=1)
        s2 = torch.stack((torch.zeros(3, 4), torch.ones(3, 4)), dim=1)

        for permutation, (s_pred, s_true) in product(
                ['min', 'max', 'none'],
                [(0.8 * s1, 0.5 * s1),
                 (0.8 * s1, 0.5 * s2),
                 (0.8 * s2, 0.5 * s2),
                 (0.8 * s2, 0.5 * s1)],
        ):

            y_pred = {'waves': s_pred}
            y_true = {'waves': s_true}
            loss_fn = CompositeLoss(
                [{'func': L1WaveformLoss(), 'weight': 0.5},
                 {'func': L2WaveformLoss(), 'weight': 1.0}],
                permutation=permutation,
                reduction='none',
            )
            loss = loss_fn(y_pred, y_true)

            if permutation == 'none':
                true_l1 = (s_true - s_pred).abs().mean()
                true_l2 = torch.mean((s_true - s_pred) ** 2)
            elif permutation == 'min':
                true_l1 = (0.8-0.5) * 4 / (2*4)
                true_l2 = (0.8-0.5)**2 * 4 / (2*4)
            elif permutation == 'max':
                true_l1 = (0.8+0.5) * 4 / (2*4)
                true_l2 = (0.8**2 + 0.5**2) * 4 / (2*4)

            for d in loss['sample']:
                self.assertLess(abs(d['l1_waveform'] - true_l1), 1e-3)
                self.assertLess(abs(d['l2_waveform'] - true_l2), 1e-3)
            for d in loss['batch']:
                self.assertLess(abs(d - 0.5 * true_l1 - true_l2), 1e-3)


    def test_forward_ok(self):
        y_pred = {
            'waves': torch.rand(3, 2, 8),
            'embd': torch.rand(3, 4, 3, 20)
        }
        y_true = {
            'waves': torch.rand(3, 2, 8),
            'embd': torch.rand(3, 4, 3, 2)
        }

        # 1. reduction = none
        loss_fn = CompositeLoss(
            [{'func': L1WaveformLoss(), 'weight': 1.0},
             {'func': L2WaveformLoss(), 'weight': 1.0},
             {'func': DeepClusteringLoss(), 'weight': 1.0}],
            permutation='min',
            reduction='none',
        )

        loss_dict_ref = loss_fn.forward_no_reduction(y_pred, y_true)
        loss_dict = loss_fn(y_pred, y_true)
        self.assertEqual(type(loss_dict), dict)
        self.assertIn('batch', loss_dict)
        self.assertIn('sample', loss_dict)
        self.assertTrue(
            torch.all(loss_dict['batch'] == loss_dict_ref['batch']))
        self.assertEqual(loss_dict['sample'], loss_dict_ref['sample'])

        # 2. reduction = mean
        loss_fn = CompositeLoss(
            [{'func': L1WaveformLoss(), 'weight': 1.0},
             {'func': L2WaveformLoss(), 'weight': 1.0},
             {'func': DeepClusteringLoss(), 'weight': 1.0}],
            permutation='min',
            reduction='mean',
        )

        loss_dict_ref = loss_fn.forward_no_reduction(y_pred, y_true)
        loss_dict = loss_fn(y_pred, y_true)
        self.assertEqual(loss_dict['batch'].shape, torch.Size())
        self.assertLess(
            abs(
                loss_dict['batch'].item()
                - sum(loss_dict_ref['batch'])
                / len(loss_dict_ref['batch'])
            ),
            1e-3
        )
        self.assertEqual(loss_dict['sample'], loss_dict_ref['sample'])

        # 3. reduction = sum
        loss_fn = CompositeLoss(
            [{'func': L1WaveformLoss(), 'weight': 1.0},
             {'func': L2WaveformLoss(), 'weight': 1.0},
             {'func': DeepClusteringLoss(), 'weight': 1.0}],
            permutation='min',
            reduction='sum',
        )

        loss_dict_ref = loss_fn.forward_no_reduction(y_pred, y_true)
        loss_dict = loss_fn(y_pred, y_true)
        self.assertEqual(loss_dict['batch'].shape, torch.Size())
        self.assertLess(
            abs(
                loss_dict['batch'].item()
                - sum(loss_dict_ref['batch'])
            ),
            1e-3
        )
        self.assertEqual(loss_dict['sample'], loss_dict_ref['sample'])

    def test_init_ng(self):

        # 1. invalid reduction
        is_error = False
        try:
            CompositeLoss(
                [{'func': L1WaveformLoss(), 'weight': 1.0},
                 {'func': DeepClusteringLoss(), 'weight': 1.0}],
                permutation='min',
                reduction='invalid',
            )
        except Exception:
            is_error = True
        self.assertTrue(is_error)

        # 2. invalid permutation
        is_error = False
        try:
            CompositeLoss(
                [{'func': L1WaveformLoss(), 'weight': 1.0},
                 {'func': DeepClusteringLoss(), 'weight': 1.0}],
                permutation='invalid',
                reduction='mean',
            )
        except Exception:
            is_error = True
        self.assertTrue(is_error)

        # 3. invalid components
        is_error = False
        try:
            CompositeLoss(
                [{'func': L1WaveformLoss(), 'weight': 1.0},
                 {'func': DeepClusteringLoss(), 'weight': 1.0},
                 {'func': None, 'weight': 1.0}],
                permutation='min',
                reduction='mean',
            )
        except Exception:
            is_error = True
        self.assertTrue(is_error)

    def test_forward_ng(self):
        # this function requires both waves and embd
        loss_fn = CompositeLoss(
            [{'func': L1WaveformLoss(), 'weight': 1.0},
             {'func': L2WaveformLoss(), 'weight': 1.0},
             {'func': DeepClusteringLoss(), 'weight': 1.0}],
            permutation='min',
            reduction='none',
        )

        y_pred = {
            'embd': torch.rand(3, 4, 3, 20),
        }
        y_true = {
            'waves': torch.rand(3, 2, 8),
            'embd': torch.rand(3, 4, 3, 2),
        }

        is_error = False
        try:
            loss_fn(y_pred, y_true)
        except:
            is_error = True
        self.assertTrue(is_error)

        y_pred = {
            'waves': torch.rand(3, 2, 8),
            'embd': torch.rand(3, 4, 3, 20),
        }
        y_true = {
            'embd': torch.rand(3, 4, 3, 2),
        }

        is_error = False
        try:
            loss_fn(y_pred, y_true)
        except:
            is_error = True
        self.assertTrue(is_error)

