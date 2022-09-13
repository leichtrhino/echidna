
import unittest
import os
import json
import math

from echidna.data import datanodes

class ToyDataNode(datanodes.DataNode):
    def __init__(self, name, children=None):
        super().__init__(children)
        self.name = name
        self.channels = list(range(8))

    def process_partial(self, data=None):
        if self.super_type == 'source':
            if data is None:
                data = list(range(8))
            return [self.name for i in data]
        else:
            r_i = max(
                range(len(data)),
                key=lambda i: 0 if data[i] is None else i
            )
            return [d+self.name if d else None for d in data][:r_i+1]

    def require_channel_index_partial(self):
        if self.name.startswith('R') or self.name.startswith('A'):
            return None
        elif self.name.startswith('M'):
            return [1, 2]
        else:
            return [5, 4, 3]

    @property
    def super_type(self):
        if self.name.startswith('R'):
            return 'empty'
        elif self.name.startswith('S') or self.name.startswith('A'):
            return 'source'
        else:
            return 'filter'

    @classmethod
    def from_dict_args(cls, d : dict, c : dict):
        return cls(d['name'])

    def to_dict_args(self):
        return {
            'name': self.name
        }

datanodes.add_datanode_cls('toy', ToyDataNode)

class TestDataNodes(unittest.TestCase):
    def test_sample(self):
        # build tree
        #      R------
        #    /   \    \
        #   S1    A1   A2----
        #  /  \       /  \   \
        # M1   M2    M3   M4  M5
        f = ToyDataNode('R', [
            ToyDataNode('S1', [
                ToyDataNode('M1'),
                ToyDataNode('M2'),
            ]),
            ToyDataNode('A1'),
            ToyDataNode('A2', [
                ToyDataNode('M3'),
                ToyDataNode('M4'),
                ToyDataNode('M5'),
            ]),
        ])

        # count leaf node
        self.assertEqual(len(f), 6)

        # enumerate leaf nodes
        for i, names in enumerate([
                ['S1', 'M1'],
                ['S1', 'M2'],
                ['A1'],
                ['A2', 'M3'],
                ['A2', 'M4'],
                ['A2', 'M5'],
        ]):
            self.assertEqual([n.name for n in f.get_single_chain(i)], names)

        # list leaf node
        self.assertEqual(len(f.list_leaf_node()), 6)
        for i, (name, node) in enumerate(zip(
                ['M1', 'M2', 'A1', 'M3', 'M4', 'M5'],
                f.list_leaf_node()
        )):
            self.assertEqual(name, node.name)

        # check filter chain
        for i, (d, md) in enumerate([
                ([None, None, None, 'S1M1', 'S1M1',], ['S1', 'M1']),
                ([None, None, None, 'S1M2', 'S1M2',], ['S1', 'M2']),
                (['A1', 'A1', 'A1', 'A1', 'A1', 'A1', 'A1', 'A1',], ['A1']),
                ([None, 'A2M3', 'A2M3',], ['A2', 'M3']),
                ([None, 'A2M4', 'A2M4',], ['A2', 'M4']),
                ([None, 'A2M5', 'A2M5',], ['A2', 'M5']),
        ]):
            data, metadata = f.process(i)
            self.assertEqual(data, d)
            self.assertEqual([m.name for m in metadata], md)

        # check metrics
        f.list_leaf_node()[0].push_metric('metric', 1.0)
        self.assertEqual(f.list_leaf_node()[0].get_metric('metric'), 1.0)

        # check serialize/deserialize
        f_dict = f.to_dict()
        f_obj = datanodes.DataNode.from_dict(f_dict)
        self.assertEqual(f_dict, f_obj.to_dict())

        # check deserialized metrics
        self.assertEqual(f_obj.list_leaf_node()[0].get_metric('metric'), 1.0)

        # balance tree by remove
        f.balance_by_remove()
        self.assertEqual(len(f), 5)
        for i, (d, md) in enumerate([
                ([None, None, None, 'S1M1', 'S1M1',], ['S1', 'M1']),
                ([None, None, None, 'S1M2', 'S1M2',], ['S1', 'M2']),
                ([None, 'A2M3', 'A2M3',], ['A2', 'M3']),
                ([None, 'A2M4', 'A2M4',], ['A2', 'M4']),
                ([None, 'A2M5', 'A2M5',], ['A2', 'M5']),
        ]):
            data, metadata = f.process(i)
            self.assertEqual(data, d)
            self.assertEqual([m.name for m in metadata], md)

