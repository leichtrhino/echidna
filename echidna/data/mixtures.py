
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
            sample_index=d['sampleIndex'],
            mixture_index=d['mixtureIndex'],
            mixture_indices=d['mixtureIndices'])

    @classmethod
    def from_list(cls, d : tp.List):
        return [cls.from_dict(d_) for d_ in d]

    def to_dict(self):
        return {
            'sampleIndex': self.sample_index,
            'mixtureIndex': self.mixture_index,
            'mixtureIndices': self.mixture_indices
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
            created_at=datetime.fromisoformat(d['createdAt']),
            seed=d['seed'],
            algorithm_out=d.get('algorithmOut', None),
        )

    def to_dict(self):
        return {
            'mixture': self.mixture.to_dict(),
            'createdAt': self.created_at.isoformat(),
            'seed': self.seed,
            'algorithmOut': self.algorithm_out,
        }

class MixturesJournal(object):
    def __init__(self,
                 process_start : datetime,
                 process_finish : datetime,
                 metadata_path : str,
                 spec,
                 mixture_journals : tp.List[MixtureJournal]):
        self.process_start = process_start
        self.process_finish = process_finish
        self.metadata_path = metadata_path
        self.spec = spec
        self.mixture_journals = mixture_journals

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            process_start=datetime.fromisoformat(d['processStart']),
            process_finish=datetime.fromisoformat(d['processFinish']),
            metadata_path=d['metadataPath'],
            spec=MixtureSpec.from_dict(d['spec']),
            mixture_journals=[
                MixtureJournal.from_dict(j)
                for j in d['mixtureJournals']
            ]
        )

    def to_dict(self):
        return {
            'processStart': self.process_start.isoformat(),
            'processFinish': self.process_finish.isoformat(),
            'metadataPath': str(self.metadata_path) if self.metadata_path else None,
            'spec': self.spec.to_dict(),
            'mixtureJournals': [j.to_dict() for j in self.mixture_journals]
        }



class MixtureSpec(object):
    def __init__(self,
                 algorithm_name : str,
                 algorithm_params : tp.Dict[str, str],
                 seed : int,
                 mix_per_sample : int,
                 sample_metadata_path : str,
                 mixture_metadata_path : str,
                 journal_path : str,
                 jobs : int=None):
        self.algorithm_name = algorithm_name
        self.algorithm_params = algorithm_params
        self.seed = seed
        self.mix_per_sample = mix_per_sample
        self.sample_metadata_path = sample_metadata_path
        self.mixture_metadata_path = mixture_metadata_path
        self.journal_path = journal_path
        self.jobs = jobs

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            algorithm_name=d['algorithmName'],
            algorithm_params=d['algorithmParams'],
            seed=d['seed'],
            mix_per_sample=d['mixPerSample'],
            sample_metadata_path=d['sampleMetadataPath'],
            mixture_metadata_path=d['mixtureMetadataPath'],
            journal_path=d['journalPath'],
            jobs=d.get('jobs', None)
        )

    def to_dict(self):
        return {
            'algorithmName': self.algorithm_name,
            'algorithmParams': self.algorithm_params,
            'seed': self.seed,
            'mixPerSample': self.mix_per_sample,
            'sampleMetadataPath': str(self.sample_metadata_path)
            if self.sample_metadata_path else None,
            'mixtureMetadataPath': str(self.mixture_metadata_path)
            if self.mixture_metadata_path else None,
            'journalPath': str(self.journal_path)
            if self.journal_path else None,
            'jobs': self.jobs
        }

    def save_mixture(self):
        _save_mixture(self)

class MixAlgorithm(object):
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

def get_mix_algorithm(name : str):
    if name not in _mix_algorithms:
        raise ValueError(f'{name} is not registered as a mix algorithm')
    return _mix_algorithms[name]

_mix_algorithms = dict()
if len(_mix_algorithms) == 0:
    register_mix_algorithm('CategoryMix', CategoryMix)


def _save_mixture(spec : MixtureSpec):
    """
    """

    process_start = datetime.now()
    # setup algorithm
    alg_cls = _mix_algorithms.get(spec.algorithm_name)
    algorithm = alg_cls(**spec.algorithm_params)

    random_ = random.Random(spec.seed)

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

    # iterate over dataset and find mixtures
    mixture_list = []
    journal_list = []
    for mixture, journal in map_fn(_make_single_mixture, args):
        mixture_list.append(mixture)
        journal_list.append(journal)

    # close map function
    if spec.jobs is not None:
        pool.close()

    process_finish = datetime.now()

    # save metadata
    if not os.path.exists(os.path.dirname(spec.mixture_metadata_path)):
        os.makedirs(os.path.dirname(spec.mixture_metadata_path))
    with open(spec.mixture_metadata_path, 'w') as fp:
        json.dump([m.to_dict() for m in mixture_list], fp)

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
            spec=spec,
            mixture_journals=journal_list,
        )
        with open(spec.journal_path, 'w') as fp:
            json.dump(mixtures_journal.to_dict(), fp)

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

