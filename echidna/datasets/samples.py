
import os
import typing as tp
import math
import json
import random
import multiprocessing
from datetime import datetime

import torch
import torchaudio
import numpy

from .transforms import Resample
from .utils import merge_activation

class Datasource(object):
    """
    Metadata for single raw wave and sheet pair

    Attributes
    ----------
    id : str
    wave_path : str
    sheet_path : str, optional
    category : str, optional
    track : str, optional
    fold : str, optional
    """
    def __init__(self,
                 id : str,
                 wave_path : str,
                 sheet_path : str,
                 category : str,
                 track : str=None,
                 fold : str=None):
        self.id = id
        self.wave_path = wave_path
        self.sheet_path = sheet_path
        self.category = category
        self.track = track
        self.fold = fold

    @classmethod
    def from_dict(cls, d : tp.Dict):
        """
        """
        return cls(id=d['id'],
                   wave_path=d['wavePath'],
                   sheet_path=d.get('sheetPath', None),
                   category=d.get('category', None),
                   track=d.get('track', None),
                   fold=d.get('fold', None))

    def to_dict(self):
        """
        """
        return {
            'id': self.id,
            'wavePath': str(self.wave_path) if self.wave_path else None,
            'sheetPath': str(self.sheet_path) if self.sheet_path else None,
            'category': self.category,
            'track': self.track,
            'fold': self.fold,
        }

class Sample(object):
    def __init__(self,
                 path : str,
                 categories : tp.List[str],
                 tracks : tp.List[str],
                 folds : str,
                 sample_rate : int):
        self.path = path
        self.categories = categories
        self.tracks = tracks
        self.folds = folds
        self.sample_rate = sample_rate

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            path=d['path'],
            categories=d['categories'],
            tracks=d.get('tracks', [None] * len(d['categories'])),
            folds=d.get('folds', [None] * len(d['categories'])),
            sample_rate=d['sampleRate'],
        )

    @classmethod
    def from_list(cls, l : tp.List):
        return [cls.from_dict(d) for d in l]

    def to_dict(self):
        return {
            'path': str(self.path) if self.path else None,
            'categories': self.categories,
            'tracks': self.tracks,
            'folds': self.folds,
            'sampleRate': self.sample_rate,
        }

class SampleSpec(object):
    """
    Metadata for datasource sampling

    Attributes
    ----------
    datasources : list
        list of Datasource objects
    sample_size : int
    source_per_category : int
    sample_rate : int
    duration : float
    seed : int
    """

    def __init__(self,
                 datasources : tp.List[Datasource],
                 fold : str,
                 sample_size : int,
                 source_per_category : int,
                 sample_rate : int,
                 duration : float,
                 seed : int,
                 metadata_path : str,
                 data_dir : str,
                 journal_path : str,
                 jobs : int=None):

        self.datasources = datasources
        self.fold = fold
        self.sample_size = sample_size
        self.source_per_category = source_per_category
        self.sample_rate = sample_rate
        self.duration = duration
        self.seed = seed
        self.metadata_path = metadata_path
        self.data_dir = data_dir
        self.journal_path = journal_path
        self.jobs = jobs

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            datasources=[Datasource.from_dict(s) for s in d['datasources']],
            fold=d['fold'],
            sample_size=d['sampleSize'],
            source_per_category=d['sourcePerCategory'],
            sample_rate=d['sampleRate'],
            duration=d['duration'],
            seed=d['seed'],
            metadata_path=d['metadataPath'],
            data_dir=d['dataDir'],
            journal_path=d.get('journalPath', None),
            jobs=d.get('jobs', None),
        )

    def to_dict(self):
        return {
            'datasources': [s.to_dict() for s in self.datasources],
            'fold': self.fold,
            'sampleSize': self.sample_size,
            'sourcePerCategory': self.source_per_category,
            'sampleRate': self.sample_rate,
            'duration': self.duration,
            'seed': self.seed,
            'metadataPath': str(self.metadata_path) if self.metadata_path else None,
            'dataDir': str(self.data_dir) if self.data_dir else None,
            'journalPath': str(self.journal_path) if self.journal_path else None,
            'jobs': self.jobs,
        }

    def save_samples(self):
        _save_sample(self)

class SampleJournal(object):
    """
    Journal class for a sample

    Attributes
    ----------
    created_at : datetime
    seed : int
    datasources : tp.List[tp.List[str]]
    length : int
    offsets : tp.List[int]
    sample : Sample)
    """

    def __init__(self,
                 created_at : datetime,
                 seed : int,
                 datasources : tp.List[tp.List[str]],
                 length : int,
                 offsets : tp.List[int],
                 sample : Sample):
        self.created_at = created_at
        self.seed = seed
        self.datasources = datasources
        self.offsets = offsets
        self.length = length
        self.sample = sample

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            created_at=datetime.fromisoformat(d['createdAt']),
            seed=d['seed'],
            datasources=d.get('datasources', []),
            length=d['length'],
            offsets=d['offsets'],
            sample=Sample.from_dict(d['sample'])
        )

    def to_dict(self):
        return {
            'createdAt': self.created_at.isoformat(),
            'seed': self.seed,
            'datasources': self.datasources,
            'length': self.length,
            'offsets': self.offsets,
            'sample': self.sample.to_dict(),
        }

class SamplesJournal(object):
    """
    Journal class of samples

    Attributes
    ----------
    process_start : timestamp
    process_finish : timestamp
    metadata_path : str
    spec : SampleSpecification
    seed : int
    samples : tp.List[SampleJournal])

    """
    def __init__(self,
                 process_start : datetime,
                 process_finish : datetime,
                 metadata_path : str,
                 spec,
                 seed : int,
                 sample_journals : tp.List[SampleJournal]):
        self.process_start = process_start
        self.process_finish = process_finish
        self.metadata_path = metadata_path
        self.spec = spec
        self.seed = seed
        self.sample_journals = sample_journals

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            process_start=datetime.fromisoformat(d['processStart']),
            process_finish=datetime.fromisoformat(d['processFinish']),
            metadata_path=d['metadataPath'],
            seed=d['seed'],
            spec=SampleSpec.from_dict(d['spec']),
            sample_journals=[
                SampleJournal.from_dict(j)
                for j in d['sampleJournals']
            ],
        )

    def to_dict(self):
        return {
            'processStart': self.process_start.isoformat(),
            'processFinish': self.process_finish.isoformat(),
            'metadataPath': str(self.metadata_path) if self.metadata_path else None,
            'seed': self.seed,
            'spec': self.spec.to_dict(),
            'sampleJournals': [j.to_dict() for j in self.sample_journals]
        }

def _save_sample(spec : SampleSpec):

