
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

    process_start = datetime.now()
    # setup algorithm
    alg_cls = _augmentation_algorithms.get(spec.algorithm_name)
    algorithm = alg_cls(**spec.algorithm_params)

    random_ = random.Random(spec.seed)

    # load metadata
    with open(spec.sample_metadata_path, 'r') as fp:
        metadata_list = Sample.from_list(json.load(fp))

    # prepare arguments
    args = [
        (
            sample_i,
            augment_i,
            algorithm,
            os.path.join(
                os.path.dirname(spec.sample_metadata_path),
                metadata.path
            ), #data_path,
            metadata,
            random_.randrange(2**32), #seed,
        )
        for sample_i, metadata in enumerate(metadata_list)
        for augment_i in range(spec.augmentation_per_sample)
    ]

    # map func
    if spec.jobs is not None:
        pool = multiprocessing.Pool(spec.jobs)
        map_fn = pool.imap_unordered
    else:
        map_fn = map

    # iterate over dataset and find mixtures
    augmentation_list = []
    journal_list = []
    for augmentation, journal in map_fn(_make_single_augmentation, args):
        augmentation_list.append(augmentation)
        journal_list.append(journal)

    # close map function
    if spec.jobs is not None:
        pool.close()

    process_finish = datetime.now()

    # save metadata
    if not os.path.exists(os.path.dirname(spec.augmentation_metadata_path)):
        os.makedirs(os.path.dirname(spec.augmentation_metadata_path))
    with open(spec.augmentation_metadata_path, 'w') as fp:
        json.dump([a.to_dict() for a in augmentation_list], fp)

    # save journal
    if spec.journal_path is not None:
        if not os.path.exists(os.path.dirname(spec.journal_path)):
            os.makedirs(os.path.dirname(spec.journal_path))
        augmentations_journal = AugmentationsJournal(
            process_start=process_start,
            process_finish=process_finish,
            metadata_path=os.path.relpath(
                spec.augmentation_metadata_path,
                os.path.dirname(spec.journal_path)
            ),
            spec=spec,
            augmentation_journals=journal_list,
        )
        with open(spec.journal_path, 'w') as fp:
            json.dump(augmentations_journal.to_dict(), fp)

def _make_single_augmentation(args):
    sample_i, augment_i, algorithm, data_path, metadata, seed \
        = args
    data = torch.load(data_path)

    transform_param, aux_out = algorithm.augmentation_params(data,
                                                             metadata,
                                                             seed)
    augmentation = Augmentation(sample_index=sample_i,
                                augmentation_index=augment_i,
                                **transform_param)
    journal = AugmentationJournal(augmentation=augmentation,
                                  created_at=datetime.now(),
                                  seed=seed,
                                  algorithm_out=aux_out)

    return augmentation, journal


def _make_param_set(data : tp.Dict[str, torch.Tensor],
                    metadata : tp.List[tp.Dict[str, object]],
                    seed : int,

                    source_sample_rate : int,
                    target_sample_rate : int,
                    waveform_length : int,

                    normalize : bool,
                    scale_range : tp.Tuple[int],
                    scale_point_range : tp.Tuple[int],
                    time_stretch_range : tp.Tuple[int],
                    pitch_shift_range : tp.Tuple[int],

                    n_fft : int=2048,
                    hop_length : int=512,
                    win_length : int=2048,
                    ) -> tp.List[tp.Dict[str, object]]:

    assert len(scale_range) == 2
    assert len(scale_point_range) == 2
    assert len(time_stretch_range) == 2
    assert len(pitch_shift_range) == 2

    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)

    # make time-dependent parameter for each track
    track_param = dict()
    track_activation = dict()
    for track in set(track for track in metadata.tracks if track is not None):
        track_param[track] = {
            'time_stretch_rate': random.uniform(*time_stretch_range),
        }
        track_activation[track] = [(0, None, [])]

    waves = data['waves']
    # make transform parameter without offset
    channel_params = []
    channel_activations = []
    transformed_waves = []
    common_params = {
        'normalize': normalize,
        'source_sample_rate': source_sample_rate,
        'target_sample_rate': target_sample_rate,
        'waveform_length': None, # this param is overwritten at the end
        'n_fft': n_fft,
        'hop_length': hop_length,
        'win_length': win_length,
    }

    for channel_i, (wave, track) in enumerate(zip(waves, metadata.tracks)):
        # build transform
        time_stretch_rate = track_param.get(
            track,
            {'time_stretch_rate' : random.uniform(*time_stretch_range)}
        )['time_stretch_rate']
        pitch_shift_rate = random.uniform(*pitch_shift_range)
        scale_points = random.randint(*scale_point_range)
        scale_amounts = [
            random.uniform(*scale_range) for _ in range(scale_points)
        ]
        scale_fractions = [
            random.random() for _ in range(scale_points-1)
        ]

        params = {
            'time_stretch_rate': time_stretch_rate,
            'pitch_shift_rate': pitch_shift_rate,
            'scale_amounts': scale_amounts,
            'scale_fractions': scale_fractions,
            'offset': None,
        }

        tf = build_transform(**common_params, **params)
        transformed_wave = tf(wave)
        transformed_waves.append(transformed_wave)

        # find activatio
        activation = track_activation.get(
            track, [(0, transformed_wave.shape[-1], [])]
        )
        merge_activation(activation, transformed_wave, channel_i)
        channel_params.append(params)
        channel_activations.append(activation)

    # get offset
    for track, wave, param, activations in \
        zip(metadata.tracks, transformed_waves, channel_params, channel_activations):
        if 'offset' in track_param.get(track, dict()):
            offset = track_param[track]['offset']
        else: # calculate offset from activation
            weights = [
                (end-start)*(10**(len(tags)-1)) if len(tags) else 0
                for start, end, tags in activations
            ]
            activation = random.choices(activations,
                                        weights=weights, k=1)[0]

            start, end, _ = activation
            start = max(0, start - waveform_length // 4)
            end = min(wave.shape[-1], end + waveform_length // 4)
            end = max(start, end - waveform_length)
            offset = random.randint(start, end)

            if track in track_param: # assign offset to track
                track_param[track]['offset'] = offset

        param['offset'] = offset

    common_params['waveform_length'] = waveform_length
    augment_param = dict(
        **common_params,
        **{
            'offsets': [
                p['offset'] for p in channel_params],
            'time_stretch_rates': [
                p['time_stretch_rate'] for p in channel_params],
            'pitch_shift_rates': [
                p['pitch_shift_rate'] for p in channel_params],
            'scale_amount_list': [
                p['scale_amounts'] for p in channel_params],
            'scale_fraction_list': [
                p['scale_fractions'] for p in channel_params],
        }
    )
    return augment_param

