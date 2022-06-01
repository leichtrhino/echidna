
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


def _save_mixture(spec : MixtureSpec):
    """
    """


