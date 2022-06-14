
import unittest
import torch

from echidna.datasets import utils

class TestMergeActivation(unittest.TestCase):
    def test_merge_activation_create(self):
        base_list = [
            (0*16000, 25*16000, []),
        ]
        x = torch.zeros(25*16000)
        x[3*16000:6*16000] = \
            torch.sin(torch.linspace(0, 2*torch.pi, 6*16000-3*16000))
        x[9*16000:12*16000] = \
            torch.sin(torch.linspace(0, 2*torch.pi, 12*16000-9*16000))
        x[16*16000:19*16000] = \
            torch.sin(torch.linspace(0, 2*torch.pi, 19*16000-16*16000))

        utils.merge_activation(base_list, x, tag='x')

        self.assertEqual(base_list[0][0], 0)
        self.assertEqual(base_list[-1][1], 25*16000)
        self.assertEqual([b[0] for b in base_list[1:]],
                         [b[1] for b in base_list[:-1]])
        self.assertEqual([b[2] for b in base_list],
                         [[], ['x'], [], ['x'], [], ['x'], []])

    def test_merge_activation_update(self):
        base_list = [
            (0*16000, 5*16000, []),
            (5*16000, 10*16000, ['a']),
            (10*16000, 15*16000, []),
            (15*16000, 20*16000, ['a']),
            (20*16000, 25*16000, []),
        ]
        x = torch.zeros(25*16000)
        x[3*16000:6*16000] = \
            torch.sin(torch.linspace(0, 2*torch.pi, 6*16000-3*16000))
        x[9*16000:12*16000] = \
            torch.sin(torch.linspace(0, 2*torch.pi, 12*16000-9*16000))
        x[16*16000:19*16000] = \
            torch.sin(torch.linspace(0, 2*torch.pi, 19*16000-16*16000))

        y = torch.zeros(25*16000)
        y[14*16000:21*16000] = \
            torch.sin(torch.linspace(0, 2*torch.pi, 21*16000-14*16000))

        utils.merge_activation(base_list, x, tag='x')
        utils.merge_activation(base_list, y, tag='y')

        self.assertEqual(base_list[0][0], 0)
        self.assertEqual(base_list[-1][1], 25*16000)
        self.assertEqual([b[0] for b in base_list[1:]],
                         [b[1] for b in base_list[:-1]])
        self.assertEqual([b[2] for b in base_list],
                         [[], ['x'], ['a', 'x'], ['a'], ['a', 'x'], ['x'],
                          [], ['y'], ['a', 'y'], ['a', 'x', 'y'],
                          ['a', 'y'], ['y'], []])

    def test_merge_activation_boundary(self):
        base_list = [
            (0*16000, 5*16000, []),
        ]
        x = torch.zeros(5*16000)
        x[0*16000:1*16000] = \
            torch.sin(torch.linspace(0, 2*torch.pi, 1*16000-0*16000))
        x[4*16000:5*16000] = \
            torch.sin(torch.linspace(0, 2*torch.pi, 5*16000-4*16000))

        utils.merge_activation(base_list, x, tag='x')

        self.assertEqual(base_list[0][0], 0)
        self.assertEqual(base_list[-1][1], 5*16000)
        self.assertEqual([b[0] for b in base_list[1:]],
                         [b[1] for b in base_list[:-1]])
        self.assertEqual([b[2] for b in base_list],
                         [['x'], [], ['x']])

