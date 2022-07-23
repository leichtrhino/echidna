
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

from .samples import Sample

logger = logging.getLogger(__name__)

class Mixture(object):
    def __init__(self,
                 sample_index : int,
                 mixture_index : int,
                 mixture_indices : tp.List[tp.List[tp.List[int]]]):
        self.sample_index = sample_index
        self.mixture_index = mixture_index
        self.mixture_indices = mixture_indices

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            sample_index=d['sample_index'],
            mixture_index=d['mixture_index'],
            mixture_indices=d['mixture_indices'])

    @classmethod
    def from_list(cls, d : tp.List):
        return [cls.from_dict(d_) for d_ in d]

    def to_dict(self):
        return {
            'sample_index': self.sample_index,
            'mixture_index': self.mixture_index,
            'mixture_indices': self.mixture_indices
        }

class MixtureJournal(object):
    def __init__(self,
                 mixture : Mixture,
                 created_at : datetime,
                 seed : int,
                 algorithm_out : tp.Dict[str, object]):
        self.mixture = mixture
        self.created_at = created_at
        self.seed = seed
        self.algorithm_out = algorithm_out

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            mixture=Mixture.from_dict(d['mixture']),
            created_at=datetime.fromisoformat(d['created_at']),
            seed=d['seed'],
            algorithm_out=d.get('algorithm_out', None),
        )

    def to_dict(self):
        return {
            'mixture': self.mixture.to_dict(),
            'created_at': self.created_at.isoformat(),
            'seed': self.seed,
            'algorithm_out': self.algorithm_out,
        }

class MixturesJournal(object):
    def __init__(self,
                 process_start : datetime,
                 process_finish : datetime,
                 metadata_path : str,
                 log_path : str,
                 spec,
                 mixture_journals : tp.List[MixtureJournal]):
        self.process_start = process_start
        self.process_finish = process_finish
        self.metadata_path = metadata_path
        self.log_path = log_path
        self.spec = spec
        self.mixture_journals = mixture_journals

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            process_start=datetime.fromisoformat(d['process_start']),
            process_finish=datetime.fromisoformat(d['process_finish']),
            metadata_path=d['metadata_path'],
            log_path=d.get('log_path'),
            spec=MixtureSpec.from_dict(d['spec']),
            mixture_journals=[
                MixtureJournal.from_dict(j)
                for j in d['mixture_journals']
            ]
        )

    def to_dict(self):
        return {
            'process_start': self.process_start.isoformat(),
            'process_finish': self.process_finish.isoformat(),
            'metadata_path': str(self.metadata_path) if self.metadata_path else None,
            'log_path': str(self.log_path) if self.log_path else None,
            'spec': self.spec.to_dict(),
            'mixture_journals': [j.to_dict() for j in self.mixture_journals]
        }



class MixtureSpec(object):
    def __init__(self,
                 algorithm,
                 seed : int,
                 mix_per_sample : int,
                 sample_metadata_path : str,
                 mixture_metadata_path : str,
                 journal_path : str,
                 log_path : str,
                 log_level : str,
                 jobs : int=None):
        self.algorithm = algorithm
        self.seed = seed
        self.mix_per_sample = mix_per_sample
        self.sample_metadata_path = sample_metadata_path
        self.mixture_metadata_path = mixture_metadata_path
        self.journal_path = journal_path
        self.log_level = log_level
        self.log_path = log_path
        self.jobs = jobs

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            algorithm=MixAlgorithm.from_dict(d['algorithm']),
            seed=d['seed'],
            mix_per_sample=d['mix_per_sample'],
            sample_metadata_path=d['sample_metadata_path'],
            mixture_metadata_path=d['mixture_metadata_path'],
            journal_path=d['journal_path'],
            log_path=d.get('log_path'),
            log_level=d.get('log_level'),
            jobs=d.get('jobs', None)
        )

    def to_dict(self):
        return {
            'algorithm': self.algorithm.to_dict(),
            'seed': self.seed,
            'mix_per_sample': self.mix_per_sample,
            'sample_metadata_path': str(self.sample_metadata_path)
            if self.sample_metadata_path else None,
            'mixture_metadata_path': str(self.mixture_metadata_path)
            if self.mixture_metadata_path else None,
            'journal_path': str(self.journal_path)
            if self.journal_path else None,
            'log_path': str(self.log_path)
            if self.log_path else None,
            'log_level': self.log_level,
            'jobs': self.jobs
        }

    def save_mixture(self):
        _save_mixture(self)

class MixAlgorithm(object):
    def to_dict(self):
        return {
            'type': _reverse_mix_algorithms[type(self)],
            'args': self.to_dict_args(),
        }

    def to_dict_args(self):
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, d : dict):
        mx_type = d['type']
        mx_class = get_mix_algorithm(mx_type)
        return mx_class.from_dict_args(d['args'])

    def mix_index(self,
                  data : tp.Dict[str, torch.Tensor],
                  metadata : Sample,
                  seed : int,
                  ) -> tp.Tuple[tp.List[tp.List[tp.List[int]]],
                                tp.Dict[str, str]]:
        """
        # mix_samples, mix_out, mix_in
        """
        raise NotImplementedError()

class CategoryMix(MixAlgorithm):
    def __init__(self,
                 mix_category_list : tp.List[tp.Union[str, tp.List[str]]],
                 include_other : bool=True,
                 check_duplicate : bool=True):
        self.mix_categories = []
        all_categories = set()
        for i, l in enumerate(mix_category_list):
            self.mix_categories.append([])
            if type(l) == str:
                if l in all_categories:
                    logger.warning(f'category {l} at {i} duplicates')
                    if check_duplicate:
                        raise ValueError(f'category {l} at {i} duplicates')
                self.mix_categories[i].append(l)
                all_categories.add(l)
                continue
            for j, c in enumerate(l):
                if c in all_categories:
                    logger.warning(f'category {c} at {i} duplicates')
                    if check_duplicate:
                        raise ValueError(f'category {c} at {i} duplicates')
                self.mix_categories[i].append(c)
                all_categories.add(c)
        self.include_other = include_other

    def to_dict_args(self):
        return {
            'mix_category_list': self.mix_categories,
            'include_other': self.include_other,
        }

    @classmethod
    def from_dict_args(cls, d : dict):
        return cls(
            mix_category_list=d['mix_category_list'],
            include_other=d.get('include_other', True),
        )

    def mix_index(self,
                  data : tp.Dict[str, torch.Tensor],
                  metadata : Sample,
                  seed : int,
                  ) -> tp.Tuple[tp.List[tp.List[tp.List[int]]],
                                tp.Dict[str, str]]:
        """
        # mix_samples, mix_out, mix_in
        """
        base_index = [[] for _ in range(len(self.mix_categories))]
        if self.include_other:
            base_index.append([])

        for i, c in enumerate(metadata.categories):
            is_other = True
            for j, ds in enumerate(self.mix_categories):
                if c in ds:
                    base_index[j].append(i)
                    is_other = False
            if self.include_other and is_other:
                base_index[-1].append(i)

        mix_index = []
        for bi in base_index:
            comb_index = sum(
                (list(combinations(bi, r)) for r in range(1, len(bi)+1)),
                []
            )
            if len(comb_index) == 0:
                comb_index.append([])
            mix_index.append(comb_index)

        return list(product(*mix_index)), dict()

def register_mix_algorithm(name : str,
                           algorithm : tp.Type):
    _mix_algorithms[name] = algorithm
    _reverse_mix_algorithms[algorithm] = name

def get_mix_algorithm(name : str):
    if name not in _mix_algorithms:
        raise ValueError(f'{name} is not registered as a mix algorithm')
    return _mix_algorithms[name]

_mix_algorithms = {
    'category': CategoryMix
}
_reverse_mix_algorithms = dict((v, k) for k, v in _mix_algorithms.items())


def _save_mixture(spec : MixtureSpec):
    """
    """

    process_start = datetime.now()
    # setup algorithm
    algorithm = spec.algorithm
    algorithm_name = algorithm.to_dict()['type']

    random_ = random.Random(spec.seed)

    # prepare log
    logger = None
    if spec.log_path:
        if not os.path.exists(os.path.dirname(spec.log_path)):
            os.makedirs(os.path.dirname(spec.log_path))
        logger = logging.getLogger(__name__)
        logger.setLevel(spec.log_level)
        handler = logging.FileHandler(str(spec.log_path))
        handler.setFormatter(
            logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)

    # load metadata
    with open(spec.sample_metadata_path, 'r') as fp:
        metadata_list = Sample.from_list(json.load(fp))

    # prepare arguments
    args = [
        (
            sample_i,
            mix_i,
            algorithm,
            os.path.join(
                os.path.dirname(spec.sample_metadata_path),
                metadata.path
            ), #data_path,
            metadata,
            random_.randrange(2**32), #seed,
        )
        for sample_i, metadata in enumerate(metadata_list)
        for mix_i in range(spec.mix_per_sample)
    ]

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
            'mix_algorithm': algorithm_name,
            'sample_size': len(metadata_list),
            'mix_per_sample': spec.mix_per_sample,
            'seed': spec.seed,
            'jobs': spec.jobs,
        }))

    # iterate over dataset and find mixtures
    mixture_list = []
    journal_list = []
    for mixture, journal in map_fn(_make_single_mixture, args):
        mixture_list.append(mixture)
        journal_list.append(journal)
        if logger:
            logger.info(json.dumps({
                'type': 'made_mixture',
                'timestamp': datetime.now().isoformat(),
                'mix_algorithm': algorithm_name,
                'sample_index': mixture.sample_index,
                'mixture_index': mixture.mixture_index,
                'mixture_size': len(mixture.mixture_indices),
            }))

    # close map function
    if spec.jobs is not None:
        pool.close()

    process_finish = datetime.now()

    # save metadata
    if not os.path.exists(os.path.dirname(spec.mixture_metadata_path)):
        os.makedirs(os.path.dirname(spec.mixture_metadata_path))
    with open(spec.mixture_metadata_path, 'w') as fp:
        json.dump([m.to_dict() for m in mixture_list], fp)

    if logger:
        logger.info(json.dumps({
            'type': 'save_mixtures',
            'timestamp': datetime.now().isoformat(),
            'mix_algorithm': algorithm_name,
            'metadata_path': str(spec.mixture_metadata_path),
            'mixture_size': len(mixture_list),
            'mixture_sample_size': sum(
                len(m.mixture_indices) for m in mixture_list),
        }))

    # save journal
    if spec.journal_path is not None:
        if not os.path.exists(os.path.dirname(spec.journal_path)):
            os.makedirs(os.path.dirname(spec.journal_path))
        mixtures_journal = MixturesJournal(
            process_start=process_start,
            process_finish=process_finish,
            metadata_path=os.path.relpath(
                spec.mixture_metadata_path,
                os.path.dirname(spec.journal_path)
            ),
            log_path=os.path.relpath(
                spec.mixture_metadata_path,
                os.path.dirname(spec.log_path)
            ) if spec.log_path else None,
            spec=spec,
            mixture_journals=journal_list,
        )
        with open(spec.journal_path, 'w') as fp:
            json.dump(mixtures_journal.to_dict(), fp)

        if logger:
            logger.info(json.dumps({
                'type': 'save_mixtures_journal',
                'timestamp': datetime.now().isoformat(),
                'mix_algorithm': algorithm_name,
                'journal_path': str(spec.journal_path),
            }))

    # finish epoch and close log handler
    if logger:
        logger.info(json.dumps({
            'type': 'finish_mixing',
            'timestamp': datetime.now().isoformat(),
            'mix_algorithm': algorithm_name,
        }))
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()

def _make_single_mixture(args):
    sample_i, mix_i, algorithm, data_path, metadata, seed = args
    data = torch.load(data_path)

    mixture_indices, aux_out = algorithm.mix_index(data,
                                               metadata,
                                               seed)
    mixture = Mixture(sample_index=sample_i,
                      mixture_index=mix_i,
                      mixture_indices=mixture_indices)
    journal = MixtureJournal(mixture=mixture,
                             created_at=datetime.now(),
                             seed=seed,
                             algorithm_out=aux_out)

    return mixture, journal

