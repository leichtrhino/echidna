
import unittest
import torch

from echidna.metrics.loss import Loss

class ToyLoss(Loss):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)

    @property
    def domains(self):
        return ('waves',)

    def forward_no_reduction(self, y_pred, y_true):
        return torch.sum(torch.abs(y_pred - y_true), dim=(1, 2))

class EmbdToyLoss(Loss):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)

    def domains(self):
        return ('embd',)

    def forward_no_reduction(self, y_pred, y_true):
        return torch.sum(torch.abs(y_pred - y_true), dim=(1, 2))

class NGToyLoss(Loss):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)


class TestLoss(unittest.TestCase):
    def test_reduction_ok(self):
        a = torch.rand(3, 2, 4)
        b = torch.rand(3, 2, 4)

        no_reduction = ToyLoss('none')(a, b)
        self.assertEqual(no_reduction.shape, (3,))

        mean_reduction = ToyLoss('mean')(a, b)
        self.assertEqual(mean_reduction.shape, torch.Size())
        self.assertLess(torch.abs(no_reduction.mean() - mean_reduction), 1e-3)

        sum_reduction = ToyLoss('sum')(a, b)
        self.assertEqual(sum_reduction.shape, torch.Size())
        self.assertLess(torch.abs(no_reduction.sum() - sum_reduction), 1e-3)

    def test_domains_ok(self):
        self.assertEqual(ToyLoss('none').domains, ('waves',))

    def test_serialize_ok(self):
        ld = Loss('none').to_dict()
        self.assertEqual(ld, {'type': 'loss', 'args': {'reduction': 'none'}})
        l = Loss.from_dict(ld)
        self.assertEqual(type(l), Loss)

    def test_reduction_ng(self):
        a = torch.rand(3, 2, 4)
        b = torch.rand(3, 2, 4)

        is_error = False
        try:
            c = NGToyLoss()(a, b)
        except NotImplementedError:
            is_error = True
        self.assertTrue(is_error)
