
import typing as tp
import os
import json
import torch

from .datanodes import DataNode, EmptyNode

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 metadata_path_list : tp.List[str],
                 device : str='cpu'):
        self.metadata_path_list = metadata_path_list
        datanode_list = []
        for p in metadata_path_list:
            with open(p, 'r') as fp:
                datanode_list.append(DataNode.from_dict(
                    json.load(fp),
                    context={
                        'rel_path': os.path.dirname(p),
                        'device': device,
                    }
                ))
        self.rootnode = EmptyNode(children=datanode_list)

    def __len__(self):
        return len(self.rootnode)

    def __getitem__(self, idx):
        return self.rootnode[idx]

    def to_dict(self):
        return {
            'metadata_path_list': self.metadata_path_list
        }

    @classmethod
    def from_dict(cls, d : dict):
        return cls(
            metadata_path_list=d['metadata_path_list'],
            device=d.get('device') or 'cpu',
        )

