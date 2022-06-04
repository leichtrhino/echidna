
import typing as tp
import os
import json
import random
import multiprocessing
from datetime import datetime
from itertools import combinations

import torch
import torchaudio
import numpy
import librosa

from .utils import merge_activation
from .transforms import build_transform
from .samples import Sample
from .mixtures import get_mix_algorithm

class Augmentation(object):
    def __init__(self,
                 sample_index : int,
                 augmentation_index : int,

                 source_sample_rate : int,
                 target_sample_rate : int,
                 waveform_length : int,
                 normalize : bool,

                 offsets : tp.List[int],
                 time_stretch_rates : tp.List[float],
                 pitch_shift_rates : tp.List[float],
                 scale_amount_list : tp.List[tp.List[float]],
                 scale_fraction_list : tp.List[tp.List[float]],

                 n_fft : int=2048,
                 hop_length : int=512,
                 win_length : int=2048
                 ):

        self.sample_index = sample_index
        self.augmentation_index = augmentation_index
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.waveform_length = waveform_length
        self.normalize = normalize

        self.offsets = offsets
        self.time_stretch_rates = time_stretch_rates
        self.pitch_shift_rates = pitch_shift_rates
        self.scale_amount_list = scale_amount_list
        self.scale_fraction_list = scale_fraction_list

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    @classmethod
    def from_list(cls, l : tp.List):
        return [cls.from_dict(d) for d in l]

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            sample_index=d['sampleIndex'],
            augmentation_index=d['augmentationIndex'],
            source_sample_rate=d['sourceSampleRate'],
            target_sample_rate=d['targetSampleRate'],
            waveform_length=d['waveformLength'],
            normalize=d.get('normalize', True),

            offsets=d['offsets'],
            time_stretch_rates=d['timeStretchRates'],
            pitch_shift_rates=d['pitchShiftRates'],
            scale_amount_list=d['scaleAmountList'],
            scale_fraction_list=d['scaleFractionList'],

            n_fft=d.get('nFft', 2048),
            hop_length=d.get('hopLength', 512),
            win_length=d.get('winLength', 2048),
        )

    def to_dict(self):
        return {
            'sampleIndex': self.sample_index,
            'augmentationIndex': self.augmentation_index,
            'sourceSampleRate': self.source_sample_rate,
            'targetSampleRate': self.target_sample_rate,
            'waveformLength': self.waveform_length,
            'normalize': self.normalize,

            'offsets': self.offsets,
            'timeStretchRates': self.time_stretch_rates,
            'pitchShiftRates': self.pitch_shift_rates,
            'scaleAmountList': self.scale_amount_list,
            'scaleFractionList': self.scale_fraction_list,

            'nFft': self.n_fft,
            'hopLength': self.hop_length,
            'winLength': self.win_length,
        }


class AugmentationJournal(object):
    def __init__(self,
                 augmentation : Augmentation,
                 created_at : datetime,
                 seed : int,
                 algorithm_out : tp.Dict[str, object]):
        self.augmentation = augmentation
        self.created_at = created_at
        self.seed = seed
        self.algorithm_out = algorithm_out

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            augmentation=Augmentation.from_dict(d['augmentation']),
            created_at=datetime.fromisoformat(d['createdAt']),
            seed=d['seed'],
            algorithm_out=d.get('algorithmOut', None),
        )

    def to_dict(self):
        return {
            'augmentation': self.augmentation.to_dict(),
            'createdAt': self.created_at.isoformat(),
            'seed': self.seed,
            'algorithmOut': self.algorithm_out,
        }

class AugmentationsJournal(object):
    def __init__(self,
                 process_start : datetime,
                 process_finish : datetime,
                 metadata_path : str,
                 spec,
                 augmentation_journals : tp.List[AugmentationJournal]):
        self.process_start = process_start
        self.process_finish = process_finish
        self.metadata_path = metadata_path
        self.spec = spec
        self.augmentation_journals = augmentation_journals

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            process_start=datetime.fromisoformat(d['processStart']),
            process_finish=datetime.fromisoformat(d['processFinish']),
            metadata_path=d['metadataPath'],
            spec=AugmentationSpec.from_dict(d['spec']),
            augmentation_journals=[
                AugmentationJournal.from_dict(j)
                for j in d['augmentationJournals']
            ]
        )

    def to_dict(self):
        return {
            'processStart': self.process_start.isoformat(),
            'processFinish': self.process_finish.isoformat(),
            'metadataPath': str(self.metadata_path)
            if self.metadata_path else None,
            'spec': self.spec.to_dict(),
            'augmentationJournals': [
                j.to_dict() for j in self.augmentation_journals]
        }

class AugmentationSpec(object):
    def __init__(self,
                 algorithm_name : str,
                 algorithm_params : tp.Dict[str, str],
                 seed : int,
                 augmentation_per_sample : int,
                 sample_metadata_path : str,
                 augmentation_metadata_path : str,
                 journal_path : str,
                 jobs : int=None):
        self.algorithm_name = algorithm_name
        self.algorithm_params = algorithm_params
        self.seed = seed
        self.augmentation_per_sample = augmentation_per_sample
        self.sample_metadata_path = sample_metadata_path
        self.augmentation_metadata_path = augmentation_metadata_path
        self.journal_path = journal_path
        self.jobs = jobs

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            algorithm_name=d['algorithmName'],
            algorithm_params=d['algorithmParams'],
            seed=d['seed'],
            augmentation_per_sample=d['augmentationPerSample'],
            sample_metadata_path=d['sampleMetadataPath'],
            augmentation_metadata_path=d['augmentationMetadataPath'],
            journal_path=d['journalPath'],
            jobs=d.get('jobs', None)
        )

    def to_dict(self):
        return {
            'algorithmName': self.algorithm_name,
            'algorithmParams': self.algorithm_params,
            'seed': self.seed,
            'augmentationPerSample': self.augmentation_per_sample,
            'sampleMetadataPath': str(self.sample_metadata_path)
            if self.sample_metadata_path else None,
            'augmentationMetadataPath': str(self.augmentation_metadata_path)
            if self.augmentation_metadata_path else None,
            'journalPath': str(self.journal_path)
            if self.journal_path else None,
            'jobs': self.jobs
        }

    def save_augmentation(self):
        _save_augmentation(self)


class AugmentationAlgorithm(object):
    def augmentation_params(self,
                            data : tp.Dict[str, torch.Tensor],
                            metadata : tp.List[tp.Dict[str, object]],
                            seed : int
                            ) -> tp.List[tp.Dict[str, object]]:
        raise NotImplementedError()


def _save_augmentation(spec : AugmentationSpec):
    """
    """


