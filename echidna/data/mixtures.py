
import os
import typing as tp
import pathlib
import json
import random
import logging
import multiprocessing
from datetime import datetime
from itertools import product, combinations

import torch

from .utils import merge_activation
from .datanodes import DataNode, add_datanode_cls

logger = logging.getLogger(__name__)

class MixNode(DataNode):
    def __init__(self, mix_index, children=None, metrics=None):
        super().__init__(children, metrics)
        self.mix_index = mix_index

    def process_partial(self, data):
        out_data = []
        for i in self.mix_index:
            wave = sum(data[_i]['wave'] for _i in i) if len(i) > 0 \
                else torch.zeros_like(data[0]['wave'])
            sheet = None
            out_data.append({
                'wave': wave,
                'sheet': sheet,
            })

        return out_data

    @property
    def super_type(self):
        return 'filter'

    def require_channel_index_partial(self):
        return sorted(set([
            i for si in self.mix_index for i in si
        ]))

    @classmethod
    def from_dict_args(cls,
                       obj : dict,
                       context : dict=None):
        return cls(mix_index=obj['mix_index'])

    def to_dict_args(self):
        return {
            'mix_index': self.mix_index,
        }

add_datanode_cls('mixture', MixNode)

class MixJournal(object):
    def __init__(self,
                 created_at : datetime,
                 seed : int,
                 mix_list : tp.List[MixNode],
                 ):
        self.created_at = created_at
        self.seed = seed
        self.mix_list = mix_list

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            created_at=datetime.fromisoformat(d['created_at']),
            seed=d['seed'],
            mix_list=[MixNode.from_dict(m) for m in d['mix_list']],
        )

    def to_dict(self):
        return {
            'created_at': self.created_at.isoformat(),
            'seed': self.seed,
            'mix_list': [m.to_dict() for m in self.mix_list],
        }

class MixSetJournal(object):
    def __init__(self,
                 process_start : datetime,
                 process_finish : datetime,
                 metadata_path : str,
                 log_path : str,
                 spec,
                 mix_journals : tp.List[MixJournal]):
        self.process_start = process_start
        self.process_finish = process_finish
        self.metadata_path = metadata_path
        self.log_path = log_path
        self.spec = spec
        self.mix_journals = mix_journals

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            process_start=datetime.fromisoformat(d['process_start']),
            process_finish=datetime.fromisoformat(d['process_finish']),
            metadata_path=d['metadata_path'],
            log_path=d.get('log_path'),
            spec=MixSetSpec.from_dict(d['spec']),
            mix_journals=[
                MixJournal.from_dict(j) if j else None
                for j in d['mix_journals']
            ] if d.get('mix_journals') else None
        )

    def to_dict(self):
        return {
            'process_start': self.process_start.isoformat(),
            'process_finish': self.process_finish.isoformat(),
            'metadata_path': str(self.metadata_path) if self.metadata_path else None,
            'log_path': str(self.log_path) if self.log_path else None,
            'spec': self.spec.to_dict(),
            'mix_journals': [
                j.to_dict() if j else None for j in self.mix_journals
            ] if self.mix_journals else None
        }

class MixSetSpec(object):
    """
    mix_category_list : list
        [
            {
                'category': ['category1', 'category2'],
                'min_channel': 1,
                'max_channel': 3,
            },
            {
                'category': ['category1', 'category3'],
                'min_channel': 1,
                'max_channel': 3,
            },
            {
                'category': ['other'],
                'min_channel': 1,
                'max_channel': 3,
            },
        ]
    """

    def __init__(self,
                 input_metadata_path : str,
                 output_metadata_path : str,
                 mix_category_list : list,
                 mix_per_parent : int,
                 seed : int,
                 journal_path : str,
                 log_path : str,
                 log_level : str='INFO',
                 jobs : int=None,
                 device : str='cpu'):
        self.input_metadata_path = input_metadata_path
        self.output_metadata_path = output_metadata_path
        self.mix_category_list = mix_category_list
        self.mix_per_parent = mix_per_parent
        self.seed = seed
        self.journal_path = journal_path
        self.log_level = log_level
        self.log_path = log_path
        self.jobs = jobs
        self.device = device

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            input_metadata_path=d['input_metadata_path'],
            output_metadata_path=d['output_metadata_path'],
            mix_category_list=d['mix_category_list'],
            mix_per_parent=d.get('mix_per_parent', 1),
            seed=d.get('seed', 0),
            journal_path=d.get('journal_path'),
            log_level=d.get('log_level', 'INFO'),
            log_path=d.get('log_path'),
            jobs=d.get('jobs', None),
            device=d.get('device') or 'cpu',
        )

    def to_dict(self):
        return {
            'input_metadata_path': str(self.input_metadata_path)
            if self.input_metadata_path else None,
            'output_metadata_path': str(self.output_metadata_path)
            if self.output_metadata_path else None,
            'mix_category_list': self.mix_category_list,
            'mix_per_parent': self.mix_per_parent,
            'seed': self.seed,
            'journal_path': str(self.journal_path)
            if self.journal_path else None,
            'log_level': self.log_level,
            'log_path': str(self.log_path)
            if self.log_path else None,
            'jobs': self.jobs,
            'device': self.device,
        }

    def save_mixture(self):
        _save_mixture(self)


def _save_mixture(spec : MixSetSpec):
    """
    """

    process_start = datetime.now()

    # setup seed
    random_ = random.Random(spec.seed)

    # prepare log
    logger = None
    if spec.log_path:
        if not os.path.exists(os.path.dirname(spec.log_path)):
            os.makedirs(os.path.dirname(spec.log_path), exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.setLevel(spec.log_level)
        handler = logging.FileHandler(str(spec.log_path))
        handler.setFormatter(
            logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)

    # load metadata
    with open(spec.input_metadata_path, 'r') as fp:
        datanode = DataNode.from_dict(
            json.load(fp),
            context={
                'rel_path': os.path.dirname(spec.input_metadata_path),
                'device': spec.device,
            },
        )
    chain_lists = [
        datanode.get_single_chain(i) for i in range(len(datanode))
    ]

    # prepare arguments
    args = [(
        sample_i,
        chain_list,
        spec.mix_category_list,
        spec.mix_per_parent,
        random_.randrange(2**32), #seed,
    ) for sample_i, chain_list in enumerate(chain_lists)]

    # map func
    if spec.jobs is not None:
        pool = multiprocessing.Pool(spec.jobs)
        map_fn = pool.imap_unordered
    else:
        map_fn = map

    if logger:
        logger.info(json.dumps({
            'type': 'start_mixing',
            'timestamp': datetime.now().isoformat(),
            'input_sample_size': len(chain_lists),
        }))

    # iterate over dataset and find mixtures
    mixture_list = [None for _ in range(len(chain_lists))]
    journal_list = [None for _ in range(len(chain_lists))]
    for i, mixtures, journals in map_fn(_make_mixtures_for_sample, args):
        mixture_list[i] = mixtures
        journal_list[i] = journals
        if logger:
            logger.info(json.dumps({
                'type': 'made_mixture',
                'timestamp': datetime.now().isoformat(),
                'sample_index': i,
                'mixture_size': len(mixtures or []),
            }))

    # close map function
    if spec.jobs is not None:
        pool.close()

    # append all mixtures to leaf node
    leaf_nodes = datanode.list_leaf_node()
    assert len(leaf_nodes) == len(mixture_list)
    for n, m in zip(leaf_nodes, mixture_list):
        if not m:
            continue
        n.children = m
        for _m in m:
            _m.parent = n
    datanode.balance_by_remove()

    process_finish = datetime.now()

    # save metadata
    if not os.path.exists(os.path.dirname(spec.output_metadata_path)):
        os.makedirs(os.path.dirname(spec.output_metadata_path))
    with open(spec.output_metadata_path, 'w') as fp:
        json.dump(datanode.to_dict(), fp)

    if logger:
        logger.info(json.dumps({
            'type': 'save_mixtures',
            'timestamp': datetime.now().isoformat(),
            'output_path': str(spec.output_metadata_path),
            'mixture_size': len(datanode),
        }))

    # save journal
    if spec.journal_path is not None:
        if not os.path.exists(os.path.dirname(spec.journal_path)):
            os.makedirs(os.path.dirname(spec.journal_path))
        mixtures_journal = MixSetJournal(
            process_start=process_start,
            process_finish=process_finish,
            metadata_path=os.path.relpath(
                spec.output_metadata_path,
                os.path.dirname(spec.journal_path)
            ),
            log_path=os.path.relpath(
                spec.output_metadata_path,
                os.path.dirname(spec.log_path)
            ) if spec.log_path else None,
            spec=spec,
            mix_journals=journal_list,
        )
        with open(spec.journal_path, 'w') as fp:
            json.dump(mixtures_journal.to_dict(), fp)

        if logger:
            logger.info(json.dumps({
                'type': 'save_mixtures_journal',
                'timestamp': datetime.now().isoformat(),
                'journal_path': str(spec.journal_path),
            }))

    # finish epoch and close log handler
    if logger:
        logger.info(json.dumps({
            'type': 'finish_mixing',
            'timestamp': datetime.now().isoformat(),
        }))
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def _make_mixtures_for_sample(args):
    sample_i, node_list, mix_category_list, mix_per_parent, seed = args

    data, metadata = DataNode.process_single_chain(node_list)
    if metadata[0].super_type != 'source':
        logger.error(json.dumps({
            'type': 'InvalidNodeType',
            'messge': 'the first element of chain is not source',
            'sample_i': sample_i,
            'seed': seed,
            'metadata': [md.to_dict() for md in metadata]
        }))

    in_channel_list = metadata[0].channels
    all_categories = set([
        c for c_l in mix_category_list for c in c_l['category']
    ])

    # build list of mixture index
    # mix_index_list = [
    #   [        # for mixture 1
    #     [0, 1] # for output channel 1
    #     [2]    # for output channel 2
    #   ],
    #            # ... for more mixtures
    #]
    mix_index_list = []
    stack = [[]]
    while stack:
        index_list = stack.pop()
        next_channel_index = len(index_list)
        next_category_list = mix_category_list[next_channel_index]

        # filter index
        candidate_channel_indices = []
        for i in range(len(in_channel_list)):
            if any(i in i_l for i_l in index_list):
                continue # no index duplication
            if all(c != 'other'
                   and in_channel_list[i].category != c
                   or c == 'other'
                   and in_channel_list[i].category in all_categories
                   for c in next_category_list['category']
            ):
                continue # filter only valid category
            candidate_channel_indices.append(i)
        if not candidate_channel_indices:
            continue # no candidate: dead end

        # add index list
        min_channel = next_category_list.get('min_channel') or 1
        max_channel = next_category_list.get('max_channel') or\
            len(candidate_channel_indices)
        for n in range(max_channel, min_channel-1, -1):
            for i_l in list(combinations(candidate_channel_indices, n))[::-1]:
                if next_channel_index < len(mix_category_list) - 1:
                    stack.append(index_list + [i_l])
                else:
                    mix_index_list.append(index_list + [i_l])

    # evaluate activations
    _mix_index_list = []
    for out_index_list in mix_index_list:
        has_activation = True
        for in_index_list in out_index_list:
            out_wave = sum(data[i]['wave'] for i in in_index_list)
            activation = [(0, out_wave.shape[-1], [])]
            merge_activation(activation, out_wave, 'tag', top_db=30)
            if all(not tags for _, _, tags in activation):
                has_activation = False
                break
        if has_activation:
            _mix_index_list.append(out_index_list)
    mix_index_list = _mix_index_list

    # no mix index
    if not mix_index_list:
        return sample_i, None, None

    random_ = random.Random(seed)
    random_.shuffle(mix_index_list)
    if mix_per_parent is not None and mix_per_parent > len(mix_index_list):
        _mix_index_list = []
        while len(_mix_index_list) < mix_per_parent:
            _mix_index_list.extend(mix_index_list)
        mix_index_list = _mix_index_list
    mix_index_list = mix_index_list[:mix_per_parent]

    random_ = random.Random(seed)
    random_.shuffle(mix_index_list)
    mix_index_list = mix_index_list[:mix_per_parent]
    node_list = [MixNode(mix_index) for mix_index in mix_index_list]
    journal_list = MixJournal(
        created_at=datetime.now(),
        seed=seed,
        mix_list=node_list,
    )

    return sample_i, node_list, journal_list

