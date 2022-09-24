
import typing as tp
import os
import logging
import json
import random
import multiprocessing
from datetime import datetime
from itertools import combinations

import torch
import torchaudio
import numpy
import librosa

from .datanodes import DataNode

_metric_cls_map = dict()
_metric_name_map = dict()

def get_metric_cls(name):
    return _metric_cls_map[name]

def get_metric_name(cls):
    return _metric_name_map[cls]

def add_metric_cls(name, cls):
    _metric_cls_map[name] = cls
    _metric_name_map[cls] = name

class MetricSpec(object):
    def __init__(self,
                 metric_name : str,
                 input_metadata_path : str,
                 output_metadata_path : str,
                 seed : int,
                 journal_path : str,
                 log_path : str,
                 log_level : str,
                 jobs : int=None,
                 device : str='cpu'):

        self.metric_name = metric_name
        self.input_metadata_path = input_metadata_path
        self.output_metadata_path = output_metadata_path
        self.seed = seed
        self.journal_path = journal_path
        self.log_level = log_level
        self.log_path = log_path
        self.jobs = jobs
        self.device = device

    @classmethod
    def from_dict(cls, d : tp.Dict):
        metric_cls = get_metric_cls(d['type'])
        metric_args = metric_cls.additional_args(d)
        return metric_cls(
            metric_name=d['metric_name'],
            input_metadata_path=d['input_metadata_path'],
            output_metadata_path=d['output_metadata_path'],
            seed=d.get('seed') or None,
            journal_path=d['journal_path'],
            log_level=d.get('log_level') or 'INFO',
            log_path=d['log_path'],
            jobs=d.get('jobs'),
            device=d.get('device') or 'cpu',
            **metric_args
        )

    @classmethod
    def additional_args(cls, d : tp.Dict):
        return dict()

    def to_dict(self):
        return dict(
            type=get_metric_name(type(self)),
            metric_name=self.metric_name,
            input_metadata_path=str(self.input_metadata_path),
            output_metadata_path=str(self.output_metadata_path),
            seed=self.seed,
            journal_path=str(self.journal_path) if self.journal_path else None,
            log_level=self.log_level,
            log_path=str(self.log_path) if self.log_path else None,
            jobs=self.jobs,
            device=self.device,
            **(self.metric_args())
        )

    def metric_args(self):
        return dict()

    def put_single_metric(self, data, metadata):
        raise NotImplementedError()

    def put_metrics(self):
        _put_metrics(self)

class MetricJournal(object):
    def __init__(self,
                 created_at : datetime,
                 seed : int,
                 metric_value : float):
        self.created_at = created_at
        self.seed = seed
        self.metric_value = metric_value

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            created_at=datetime.fromisoformat(d['created_at']),
            seed=d['seed'],
            metric_value=d['metric_value'],
        )

    def to_dict(self):
        return {
            'created_at': self.created_at.isoformat(),
            'seed': self.seed,
            'metric_value': self.metric_value,
        }

class MetricSetJournal(object):
    def __init__(self,
                 process_start : datetime,
                 process_finish : datetime,
                 metadata_path : str,
                 log_path : str,
                 spec,
                 metric_journals : tp.List[MetricJournal]):
        self.process_start = process_start
        self.process_finish = process_finish
        self.metadata_path = metadata_path
        self.log_path = log_path
        self.spec = spec
        self.metric_journals = metric_journals

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            process_start=datetime.fromisoformat(d['process_start']),
            process_finish=datetime.fromisoformat(d['process_finish']),
            metadata_path=d['metadata_path'],
            log_path=d.get('log_path'),
            spec=MetricSpec.from_dict(d['spec']),
            metric_journals=[
                MetricJournal.from_dict(j) if j else None
                for j in d['metric_journals']
            ] if d.get('metric_journals') else None
        )

    def to_dict(self):
        return {
            'process_start': self.process_start.isoformat(),
            'process_finish': self.process_finish.isoformat(),
            'metadata_path': str(self.metadata_path) if self.metadata_path else None,
            'log_path': str(self.log_path) if self.log_path else None,
            'spec': self.spec.to_dict(),
            'metric_journals': [
                j.to_dict() if j else None for j in self.metric_journals
            ] if self.metric_journals else None
        }


class EntropyDifficulty(MetricSpec):
    def __init__(self,
                 metric_name : str,
                 input_metadata_path : str,
                 output_metadata_path : str,
                 seed : int,
                 journal_path : str,
                 log_path : str,
                 log_level : str,
                 jobs : int=None,
                 device : str='cpu',
                 n_fft : int=2048,
                 win_length : int=2048,
                 hop_length : int=512,
                 ):

        super().__init__(
            metric_name=metric_name,
            input_metadata_path=input_metadata_path,
            output_metadata_path=output_metadata_path,
            seed=seed,
            journal_path=journal_path,
            log_path=log_path,
            log_level=log_level,
            jobs=jobs,
            device=device,
        )
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

    def put_single_metric(self, data, metadata):
        specgrams = [
            torch.stft(
                d['wave'],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                return_complex=True,
            ).abs().clamp(min=1e-3) ** 2
            for d in data
        ]

        total_specgram = sum(specgrams)
        score = sum(
            torch.sum(-specgram/total_specgram
                      * torch.log2(specgram/total_specgram))
            for specgram in specgrams
        ) / total_specgram.numel()
        return score.item()

    @classmethod
    def additional_args(cls, d : tp.Dict):
        return {
            'n_fft': d.get('n_fft') or 2048,
            'win_length': d.get('win_length') or 2048,
            'hop_length': d.get('hop_length') or 512,
        }

    def metric_args(self):
        return {
            'n_fft': self.n_fft,
            'win_length': self.win_length,
            'hop_length': self.hop_length,
        }

add_metric_cls('entropy', EntropyDifficulty)

class FrequencyDifficulty(MetricSpec):
    def __init__(self,
                 metric_name : str,
                 input_metadata_path : str,
                 output_metadata_path : str,
                 seed : int,
                 sample_rate : int,
                 journal_path : str,
                 log_path : str,
                 log_level : str,
                 jobs : int=None,
                 device : str='cpu',
                 win_length : int=2048
    ):

        super().__init__(
            metric_name=metric_name,
            input_metadata_path=input_metadata_path,
            output_metadata_path=output_metadata_path,
            seed=seed,
            journal_path=journal_path,
            log_path=log_path,
            log_level=log_level,
            jobs=jobs,
            device=device,
        )
        self.sample_rate = sample_rate
        self.win_length = win_length

    def put_single_metric(self, data, metadata):
        waves = numpy.stack([d['wave'].numpy() for d in data])
        f0, voiced_flag, voiced_prob = librosa.pyin(
            waves,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate,
            frame_length=self.win_length,
        )
        f0 = numpy.nan_to_num(f0, nan=0.0)
        score = sum(
            numpy.abs(f0_a - f0_b).sum().item()
            for f0_a, f0_b in combinations(f0, 2)
        ) / (len(f0)*(len(f0)-1)/2 * f0.shape[-1])

        return score

    @classmethod
    def additional_args(cls, d : tp.Dict):
        return {
            'sample_rate': d['sample_rate'],
            'win_length': d.get('win_length') or 2048,
        }

    def metric_args(self):
        return {
            'sample_rate': self.sample_rate,
            'win_length': self.win_length,
        }

add_metric_cls('frequency', FrequencyDifficulty)

def _put_metrics(spec : MetricSpec):

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
        spec,
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
    leaf_nodes = datanode.list_leaf_node()
    journal_list = [None for _ in range(len(datanode))]
    for i, metric_value, journals in map_fn(_make_metric_for_sample, args):
        leaf_nodes[i].push_metric(spec.metric_name, metric_value)
        journal_list[i] = journals
        if logger:
            logger.info(json.dumps({
                'type': 'made_metric',
                'timestamp': datetime.now().isoformat(),
                'sample_index': i,
                'metric_name': spec.metric_name,
                'metric_value': metric_value,
            }))

    # close map function
    if spec.jobs is not None:
        pool.close()

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
            'output_size': len(datanode),
        }))

    # save journal
    if spec.journal_path is not None:
        if not os.path.exists(os.path.dirname(spec.journal_path)):
            os.makedirs(os.path.dirname(spec.journal_path))
        metric_journal = MetricSetJournal(
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
            metric_journals=journal_list,
        )
        with open(spec.journal_path, 'w') as fp:
            json.dump(metric_journal.to_dict(), fp)

        if logger:
            logger.info(json.dumps({
                'type': 'save_metric_journal',
                'timestamp': datetime.now().isoformat(),
                'journal_path': str(spec.journal_path),
            }))

    # finish epoch and close log handler
    if logger:
        logger.info(json.dumps({
            'type': 'finish_metric',
            'timestamp': datetime.now().isoformat(),
        }))
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def _make_metric_for_sample(args):
    sample_i, node_list, spec, seed = args
    data, metadata = DataNode.process_single_chain(node_list)
    metric_value = spec.put_single_metric(data, metadata)
    journal = MetricJournal(created_at=datetime.now(),
                            seed=seed,
                            metric_value=metric_value)

    return sample_i, metric_value, journal

