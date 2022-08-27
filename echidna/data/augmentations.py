
import typing as tp
import os
import logging
import json
import random
import multiprocessing
from datetime import datetime
from itertools import combinations, product

import torch
import torchaudio
import numpy
import librosa

from .utils import merge_activation
from .transforms import build_transform, Crop
from .samples import Sample
from .mixtures import MixAlgorithm

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
            sample_index=d['sample_index'],
            augmentation_index=d['augmentation_index'],
            source_sample_rate=d['source_sample_rate'],
            target_sample_rate=d['target_sample_rate'],
            waveform_length=d['waveform_length'],
            normalize=d.get('normalize', True),

            offsets=d['offsets'],
            time_stretch_rates=d['time_stretch_rates'],
            pitch_shift_rates=d['pitch_shift_rates'],
            scale_amount_list=d['scale_amount_list'],
            scale_fraction_list=d['scale_fraction_list'],

            n_fft=d.get('n_fft', 2048),
            hop_length=d.get('hop_length', 512),
            win_length=d.get('win_length', 2048),
        )

    def to_dict(self):
        return {
            'sample_index': self.sample_index,
            'augmentation_index': self.augmentation_index,
            'source_sample_rate': self.source_sample_rate,
            'target_sample_rate': self.target_sample_rate,
            'waveform_length': self.waveform_length,
            'normalize': self.normalize,

            'offsets': self.offsets,
            'time_stretch_rates': self.time_stretch_rates,
            'pitch_shift_rates': self.pitch_shift_rates,
            'scale_amount_list': self.scale_amount_list,
            'scale_fraction_list': self.scale_fraction_list,

            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
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
            created_at=datetime.fromisoformat(d['created_at']),
            seed=d['seed'],
            algorithm_out=d.get('algorithm_out', None),
        )

    def to_dict(self):
        return {
            'augmentation': self.augmentation.to_dict(),
            'created_at': self.created_at.isoformat(),
            'seed': self.seed,
            'algorithm_out': self.algorithm_out,
        }

class AugmentationsJournal(object):
    def __init__(self,
                 process_start : datetime,
                 process_finish : datetime,
                 metadata_path : str,
                 log_path : str,
                 spec,
                 augmentation_journals : tp.List[AugmentationJournal]):
        self.process_start = process_start
        self.process_finish = process_finish
        self.metadata_path = metadata_path
        self.log_path = log_path
        self.spec = spec
        self.augmentation_journals = augmentation_journals

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            process_start=datetime.fromisoformat(d['process_start']),
            process_finish=datetime.fromisoformat(d['process_finish']),
            metadata_path=d['metadata_path'],
            log_path=d.get('log_path'),
            spec=AugmentationSpec.from_dict(d['spec']),
            augmentation_journals=[
                AugmentationJournal.from_dict(j)
                for j in d['augmentation_journals']
            ]
        )

    def to_dict(self):
        return {
            'process_start': self.process_start.isoformat(),
            'process_finish': self.process_finish.isoformat(),
            'metadata_path': str(self.metadata_path)
            if self.metadata_path else None,
            'log_path': str(self.log_path)
            if self.log_path else None,
            'spec': self.spec.to_dict(),
            'augmentation_journals': [
                j.to_dict() for j in self.augmentation_journals]
        }

class AugmentationSpec(object):
    def __init__(self,
                 algorithm,
                 seed : int,
                 augmentation_per_sample : int,
                 sample_metadata_path : str,
                 augmentation_metadata_path : str,
                 journal_path : str,
                 log_path : str,
                 log_level : str,
                 jobs : int=None):
        self.algorithm = algorithm
        self.seed = seed
        self.augmentation_per_sample = augmentation_per_sample
        self.sample_metadata_path = sample_metadata_path
        self.augmentation_metadata_path = augmentation_metadata_path
        self.journal_path = journal_path
        self.log_path = log_path
        self.log_level = log_level
        self.jobs = jobs

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            algorithm=AugmentationAlgorithm.from_dict(d['algorithm']),
            seed=d['seed'],
            augmentation_per_sample=d['augmentation_per_sample'],
            sample_metadata_path=d['sample_metadata_path'],
            augmentation_metadata_path=d['augmentation_metadata_path'],
            journal_path=d['journal_path'],
            log_path=d.get('log_path'),
            log_level=d.get('log_level'),
            jobs=d.get('jobs', None)
        )

    def to_dict(self):
        return {
            'algorithm': self.algorithm.to_dict(),
            'seed': self.seed,
            'augmentation_per_sample': self.augmentation_per_sample,
            'sample_metadata_path': str(self.sample_metadata_path)
            if self.sample_metadata_path else None,
            'augmentation_metadata_path': str(self.augmentation_metadata_path)
            if self.augmentation_metadata_path else None,
            'journal_path': str(self.journal_path)
            if self.journal_path else None,
            'log_path': str(self.log_path)
            if self.log_path else None,
            'log_level': self.log_level,
            'jobs': self.jobs
        }

    def save_augmentation(self):
        _save_augmentation(self)


class AugmentationAlgorithm(object):
    def to_dict(self):
        return {
            'type': _reverse_augmentation_algorithms[type(self)],
            'args': self.to_dict_args(),
        }

    def to_dict_args(self):
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, d : dict):
        ag_type = d['type']
        ag_class = get_augmentation_algorithm(ag_type)
        return ag_class.from_dict_args(d['args'])

    def augmentation_params(self,
                            data : tp.Dict[str, torch.Tensor],
                            metadata : tp.List[tp.Dict[str, object]],
                            seed : int,
                            n_augmentations : int=1,
                            ) -> tp.List[tp.Dict[str, object]]:
        raise NotImplementedError()

class RandomAugmentation(AugmentationAlgorithm):
    def __init__(self,
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
                 ):

        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.waveform_length = waveform_length

        self.normalize = normalize
        self.scale_range = scale_range
        self.scale_point_range = scale_point_range
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def to_dict_args(self):
        return {
            'source_sample_rate': self.source_sample_rate,
            'target_sample_rate': self.target_sample_rate,
            'waveform_length': self.waveform_length,

            'normalize': self.normalize,
            'scale_range': self.scale_range,
            'scale_point_range': self.scale_point_range,
            'time_stretch_range': self.time_stretch_range,
            'pitch_shift_range': self.pitch_shift_range,

            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
        }

    @classmethod
    def from_dict_args(cls, d : dict):
        return cls(
            source_sample_rate=d['source_sample_rate'],
            target_sample_rate=d['target_sample_rate'],
            waveform_length=d['waveform_length'],

            normalize=d['normalize'],
            scale_range=d['scale_range'],
            scale_point_range=d['scale_point_range'],
            time_stretch_range=d['time_stretch_range'],
            pitch_shift_range=d['pitch_shift_range'],

            n_fft=d.get('n_fft', 2048),
            hop_length=d.get('hop_length', 512),
            win_length=d.get('win_length', 2048),
        )


    def augmentation_params(self,
                            data : tp.Dict[str, torch.Tensor],
                            metadata : tp.List[tp.Dict[str, object]],
                            seed : int,
                            n_augmentations : int=1,
                            ) -> tp.List[tp.List[tp.Dict[str, object]]]:

        augmentation_param_list = []
        for augmentation_params, _ in _make_param_set_sequential(
                data=data,
                metadata=metadata,
                seed=seed,
                n_augmentations=n_augmentations,
                source_sample_rate=self.source_sample_rate,
                target_sample_rate=self.target_sample_rate,
                waveform_length=self.waveform_length,
                normalize=self.normalize,
                scale_range=self.scale_range,
                scale_point_range=self.scale_point_range,
                time_stretch_range=self.time_stretch_range,
                pitch_shift_range=self.pitch_shift_range,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
        ):
            algorithm_out = None
            augmentation_param_list.append((
                augmentation_params, algorithm_out
            ))

        return augmentation_param_list

class EntropyAugmentation(AugmentationAlgorithm):
    def __init__(self,
                 source_sample_rate : int,
                 target_sample_rate : int,
                 waveform_length : int,

                 normalize : bool,
                 scale_range : tp.Tuple[int],
                 scale_point_range : tp.Tuple[int],
                 time_stretch_range : tp.Tuple[int],
                 pitch_shift_range : tp.Tuple[int],

                 mixture_algorithm : MixAlgorithm,
                 trials_per_augmentation : int,
                 separation_difficulty : float,
                 select_cartesian_product : bool=False,

                 n_fft : int=2048,
                 hop_length : int=512,
                 win_length : int=2048,
                 device : str='cpu',
                 ):

        assert 0.0 <= separation_difficulty <= 1.0

        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.waveform_length = waveform_length

        self.normalize = normalize
        self.scale_range = scale_range
        self.scale_point_range = scale_point_range
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range

        self.mixture_algorithm = mixture_algorithm
        self.trials_per_augmentation = trials_per_augmentation
        self.separation_difficulty = separation_difficulty
        self.select_cartesian_product = select_cartesian_product

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.device = device

    def to_dict_args(self):
        return {
            'source_sample_rate': self.source_sample_rate,
            'target_sample_rate': self.target_sample_rate,
            'waveform_length': self.waveform_length,

            'normalize': self.normalize,
            'scale_range': self.scale_range,
            'scale_point_range': self.scale_point_range,
            'time_stretch_range': self.time_stretch_range,
            'pitch_shift_range': self.pitch_shift_range,

            'mixture_algorithm': self.mixture_algorithm.to_dict(),
            'trials_per_augmentation': self.trials_per_augmentation,
            'separation_difficulty': self.separation_difficulty,
            'select_cartesian_product': self.select_cartesian_product,

            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
            'device': self.device,
        }

    @classmethod
    def from_dict_args(cls, d : dict):
        return cls(
            source_sample_rate=d['source_sample_rate'],
            target_sample_rate=d['target_sample_rate'],
            waveform_length=d['waveform_length'],

            normalize=d['normalize'],
            scale_range=d['scale_range'],
            scale_point_range=d['scale_point_range'],
            time_stretch_range=d['time_stretch_range'],
            pitch_shift_range=d['pitch_shift_range'],

            mixture_algorithm=MixAlgorithm.from_dict(d['mixture_algorithm']),
            trials_per_augmentation=d['trials_per_augmentation'],
            separation_difficulty=d['separation_difficulty'],
            select_cartesian_product=d.get('select_cartesian_product', False),

            n_fft=d.get('n_fft', 2048),
            hop_length=d.get('hop_length', 512),
            win_length=d.get('win_length', 2048),
            device=d.get('device', 'cpu'),
        )

    def augmentation_params(self,
                            data : tp.Dict[str, torch.Tensor],
                            metadata : tp.List[tp.Dict[str, object]],
                            seed : int,
                            n_augmentations : int=1,
                            ) -> tp.List[tp.Dict[str, object]]:
        # calculate mix index from data and metadata
        # using self.algorithm
        data = dict(
            (k, v.to(self.device) if type(v) == torch.Tensor else v)
            for k, v in data.items()
        )
        mixture_algorithm = self.mixture_algorithm
        mix_indices, _ = mixture_algorithm.mix_index(data=data,
                                                     metadata=metadata,
                                                     seed=seed)

        # return plain augmentation if mix_indices does not exist
        if len(mix_indices) == 0:
            n_channels = data['waves'].shape[0]
            params = {
                'normalize': self.normalize,
                'source_sample_rate': self.source_sample_rate,
                'target_sample_rate': self.target_sample_rate,
                'waveform_length': self.waveform_length,
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'win_length': self.win_length,
                'offsets': [0.0] * n_channels,
                'time_stretch_rates': [1.0] * n_channels,
                'pitch_shift_rates': [1.0] * n_channels,
                'scale_amount_list': [[1.0]] * n_channels,
                'scale_fraction_list': [[]] * n_channels,
            }
            algorithm_out = [
                (params, None) for _ in range(n_augmentations)
            ]
            return algorithm_out

        random_ = random.Random(seed)

        # check the number of permutations of augmentations
        use_cartesian_product = False
        if self.select_cartesian_product:
            total_augmentations = self.trials_per_augmentation ** (
                sum(1 for t in metadata.tracks if t is None)
                + len(set([t for t in metadata.tracks if t]))
            )
            use_cartesian_product = total_augmentations >= n_augmentations

        if use_cartesian_product:
            # make random augmentation parameter sets with cartesian product
            parameter_set_list = _make_param_set_cartesian_product(
                data=data,
                metadata=metadata,
                seed=seed,
                n_augmentations=self.trials_per_augmentation,
                source_sample_rate=self.source_sample_rate,
                target_sample_rate=self.target_sample_rate,
                waveform_length=self.waveform_length,
                normalize=self.normalize,
                scale_range=self.scale_range,
                scale_point_range=self.scale_point_range,
                time_stretch_range=self.time_stretch_range,
                pitch_shift_range=self.pitch_shift_range,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length
            )
            if len(parameter_set_list) \
               > n_augmentations * self.trials_per_augmentation**2:
                indices = list(range(len(parameter_set_list)))
                random_.shuffle(indices)
                indices = indices[
                    :n_augmentations * self.trials_per_augmentation**2]
                parameter_set_list = [parameter_set_list[i] for i in indices]

        else: # if n_augmentations > total_augmentations
            # make n_augmentations*trials_per_augmentation
            # random augmentation parameter sets
            parameter_set_list = _make_param_set_sequential(
                data=data,
                metadata=metadata,
                seed=seed,
                n_augmentations=n_augmentations \
                * self.trials_per_augmentation,
                source_sample_rate=self.source_sample_rate,
                target_sample_rate=self.target_sample_rate,
                waveform_length=self.waveform_length,
                normalize=self.normalize,
                scale_range=self.scale_range,
                scale_point_range=self.scale_point_range,
                time_stretch_range=self.time_stretch_range,
                pitch_shift_range=self.pitch_shift_range,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
            )

        # calculate spectrogram for each source in data
        # evaluate difficulty for each augmentation param. set
        evaluation_scores = []
        for param, wave in parameter_set_list:
            evaluation_scores.append(self.calculate_score(wave, mix_indices))

        # find most appropriate param. set
        if use_cartesian_product:
            # select n_augmentations parameters by full list
            sorted_scores = sorted(
                enumerate(evaluation_scores),
                key=lambda e: e[1],
                reverse=self.score_difficulty_order() == 'desc'
            )
            min_i = int(self.separation_difficulty * len(sorted_scores)) \
                - n_augmentations // 2
            if min_i < 0:
                min_i = 0
            elif min_i + n_augmentations > len(sorted_scores):
                min_i = len(sorted_scores) - n_augmentations
            max_i = min_i + n_augmentations

            param_i_list, score_list = zip(*[
                sorted_scores[i] for i in range(min_i, max_i)
            ])
            out_parameter_list = [parameter_set_list[i] for i in param_i_list]

            score_stats = {
                'min': sorted_scores[0][1],
                'median': sorted_scores[len(sorted_scores)//2][1],
                'mean': sum(s[1] for s in sorted_scores) / len(sorted_scores),
                'max': sorted_scores[-1][1],
            }
            algorithm_out = [
                # take augmentation parameter only
                (p[0], {'score': s, 'score_stats': score_stats})
                for p, s in zip(out_parameter_list, score_list)
            ]

        else:
            algorithm_out = []
            # group by trials
            for t_i in range(n_augmentations):
                l_i = t_i * self.trials_per_augmentation
                r_i = (t_i + 1) * self.trials_per_augmentation

                sorted_scores = sorted(
                    enumerate(evaluation_scores[l_i:r_i], l_i),
                    key=lambda e: e[1])
                sorted_i = int(self.separation_difficulty * len(sorted_scores))
                sorted_i = min(
                    max(sorted_i, 0),
                    self.trials_per_augmentation-1)
                param_i, score = sorted_scores[sorted_i]

                # get score statistics
                score_stats = {
                    'min': sorted_scores[0][1],
                    'median': sorted_scores[len(sorted_scores)//2][1],
                    'mean': sum(s[1] for s in sorted_scores) /
                    len(sorted_scores),
                    'max': sorted_scores[-1][1],
                }
                algorithm_out.append((
                    parameter_set_list[param_i][0], # take parameter set only
                    {'score': score, 'score_stats': score_stats}
                ))
        return algorithm_out

    def calculate_score(self,
                        wave,
                        mix_indices):
        specgrams = torch.stack([
            torch.stft(
                w,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                return_complex=True,
            ).abs().clamp(min=1e-3) ** 2
            for w in wave
        ], dim=0)

        mix_index = max(mix_indices, key=lambda m: sum(map(len, m)))
        mix_sg = [
            torch.sum(specgrams[list(mi)], dim=0)
            for mi in mix_index if len(mi) > 0
        ]
        total_sg = sum(mix_sg)
        score = sum(
            torch.sum(-sg/total_sg * torch.log2(sg/total_sg))
            for sg in mix_sg
        ) / total_sg.numel()
        return score.item()

    def score_difficulty_order(self):
        return 'asc'

class FrequencyAugmentation(EntropyAugmentation):
    def __init__(self,
                 source_sample_rate : int,
                 target_sample_rate : int,
                 waveform_length : int,

                 normalize : bool,
                 scale_range : tp.Tuple[int],
                 scale_point_range : tp.Tuple[int],
                 time_stretch_range : tp.Tuple[int],
                 pitch_shift_range : tp.Tuple[int],

                 mixture_algorithm : MixAlgorithm,
                 trials_per_augmentation : int,
                 separation_difficulty : float,
                 select_cartesian_product : bool=False,

                 n_fft : int=2048,
                 hop_length : int=512,
                 win_length : int=2048,
                 ):

        super().__init__(
            source_sample_rate=source_sample_rate,
            target_sample_rate=target_sample_rate,
            waveform_length=waveform_length,

            normalize=normalize,
            scale_range=scale_range,
            scale_point_range=scale_point_range,
            time_stretch_range=time_stretch_range,
            pitch_shift_range=pitch_shift_range,

            mixture_algorithm=mixture_algorithm,
            trials_per_augmentation=trials_per_augmentation,
            separation_difficulty=separation_difficulty,
            select_cartesian_product=select_cartesian_product,

            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            device='cpu',
        )

    def to_dict_args(self):
        return {
            'source_sample_rate': self.source_sample_rate,
            'target_sample_rate': self.target_sample_rate,
            'waveform_length': self.waveform_length,

            'normalize': self.normalize,
            'scale_range': self.scale_range,
            'scale_point_range': self.scale_point_range,
            'time_stretch_range': self.time_stretch_range,
            'pitch_shift_range': self.pitch_shift_range,

            'mixture_algorithm': self.mixture_algorithm.to_dict(),
            'trials_per_augmentation': self.trials_per_augmentation,
            'separation_difficulty': self.separation_difficulty,
            'select_cartesian_product': self.select_cartesian_product,

            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
        }

    @classmethod
    def from_dict_args(cls, d : dict):
        return cls(
            source_sample_rate=d['source_sample_rate'],
            target_sample_rate=d['target_sample_rate'],
            waveform_length=d['waveform_length'],

            normalize=d['normalize'],
            scale_range=d['scale_range'],
            scale_point_range=d['scale_point_range'],
            time_stretch_range=d['time_stretch_range'],
            pitch_shift_range=d['pitch_shift_range'],

            mixture_algorithm=MixAlgorithm.from_dict(d['mixture_algorithm']),
            trials_per_augmentation=d['trials_per_augmentation'],
            separation_difficulty=d['separation_difficulty'],
            select_cartesian_product=d.get('select_cartesian_product', False),

            n_fft=d.get('n_fft', 2048),
            hop_length=d.get('hop_length', 512),
            win_length=d.get('win_length', 2048),
        )

    def calculate_score(self,
                        waves,
                        mix_indices):

        if type(waves) == list:
            waves = torch.stack(waves)
        waves = waves.numpy()

        mix_index = max(mix_indices, key=lambda m: sum(map(len, m)))
        mix_waves = numpy.stack([
            waves[list(mi), :].sum(axis=0)
            for mi in mix_index if len(mi) > 0
        ])
        f0, voiced_flag, voiced_prob = librosa.pyin(
            mix_waves,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.target_sample_rate,
            frame_length=self.win_length,
        )
        f0 = numpy.nan_to_num(f0, nan=0.0)
        score = sum(
            numpy.abs(f0_a - f0_b).sum().item()
            for f0_a, f0_b in combinations(f0, 2)
        ) / (len(f0)*(len(f0)-1)/2 * f0.shape[-1])

        return score

    def score_difficulty_order(self):
        return 'desc'


def get_augmentation_algorithm(name : str):
    if name not in _augmentation_algorithms:
        raise ValueError(f'{name} is invalid augmentation algorithm')
    return _augmentation_algorithms[name]

def register_augmentation_algorithm(name : str,
                                    algorithm : tp.Type):
    _augmentation_algorithms[name] = algorithm
    _reverse_augmentation_algorithms[algorithm] = name

_augmentation_algorithms = {
    'random': RandomAugmentation,
    'entropy': EntropyAugmentation,
    'frequency': FrequencyAugmentation,
}
_reverse_augmentation_algorithms = dict(
    (v, k) for k, v in _augmentation_algorithms.items())

def _save_augmentation(spec : AugmentationSpec):
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
            os.makedirs(os.path.dirname(spec.log_path), exist_ok=True)
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
            spec.augmentation_per_sample, # augment_n
            algorithm,
            os.path.join(
                os.path.dirname(spec.sample_metadata_path),
                metadata.path
            ), #data_path,
            metadata,
            random_.randrange(2**32), #seed,
        )
        for sample_i, metadata in enumerate(metadata_list)
    ]

    # map func
    if spec.jobs is not None:
        pool = multiprocessing.Pool(spec.jobs)
        map_fn = pool.imap_unordered
    else:
        map_fn = map

    if logger:
        logger.info(json.dumps({
            'type': 'start_augmentation',
            'timestamp': datetime.now().isoformat(),
            'augmentation_algorithm': algorithm_name,
            'sample_size': len(metadata_list),
            'augmentation_per_sample': spec.augmentation_per_sample,
            'seed': spec.seed,
            'jobs': spec.jobs,
        }))

    # iterate over dataset and find mixtures
    augmentation_list = []
    journal_list = []
    for augmentations, journals in map_fn(_make_augmentations_for_sample, args):
        augmentation_list.extend(augmentations)
        journal_list.extend(journals)
        if logger:
            for augmentation in augmentations:
                logger.info(json.dumps({
                    'type': 'made_augmentation',
                    'timestamp': datetime.now().isoformat(),
                    'augmentation_algorithm': algorithm_name,
                    'sample_index': augmentation.sample_index,
                    'augmentation_index': augmentation.augmentation_index,
                }))

    # close map function
    if spec.jobs is not None:
        pool.close()

    process_finish = datetime.now()

    if logger:
        logger.info(json.dumps({
            'type': 'save_augmentations',
            'timestamp': datetime.now().isoformat(),
            'augmentation_algorithm': algorithm_name,
            'metadata_path': str(spec.augmentation_metadata_path),
            'augmentation_size': len(augmentation_list),
        }))

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
            log_path=os.path.relpath(
                spec.log_path,
                os.path.dirname(spec.log_path)
            ) if spec.log_path else None,
            spec=spec,
            augmentation_journals=journal_list,
        )
        with open(spec.journal_path, 'w') as fp:
            json.dump(augmentations_journal.to_dict(), fp)

        if logger:
            logger.info(json.dumps({
                'type': 'save_augmentations_journal',
                'timestamp': datetime.now().isoformat(),
                'augmentation_algorithm': algorithm_name,
                'journal_path': str(spec.journal_path),
            }))

    # finish epoch and close log handler
    if logger:
        logger.info(json.dumps({
            'type': 'finish_augmentation',
            'timestamp': datetime.now().isoformat(),
            'augmentation_algorithm': algorithm_name,
        }))
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()




def _make_augmentations_for_sample(args):
    sample_i, augment_n, algorithm, data_path, metadata, seed \
        = args
    data = torch.load(data_path)

    augmentations = []
    journals = []
    for augment_i, (transform_param, aux_out) in \
        enumerate(algorithm.augmentation_params(
            data,
            metadata,
            seed,
            augment_n)):
        if augment_i >= augment_n:
            break
        augmentation = Augmentation(sample_index=sample_i,
                                    augmentation_index=augment_i,
                                    **transform_param)
        journal = AugmentationJournal(augmentation=augmentation,
                                      created_at=datetime.now(),
                                      seed=seed,
                                      algorithm_out=aux_out)
        augmentations.append(augmentation)
        journals.append(journal)

    return augmentations, journals

def _make_param_set_sequential(
        data : tp.Dict[str, torch.Tensor],
        metadata : tp.List[tp.Dict[str, object]],
        seed : int,
        n_augmentations : int,

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
):

    random_ = random.Random(seed)
    seeds = [
        random_.randrange(2**32)
        for _ in range(n_augmentations)
    ]

    augmentation_param_list = []
    for seed_ in seeds:
        augmentation_params, waves = _make_param_set(
            data=data,
            metadata=metadata,
            seed=seed_,
            source_sample_rate=source_sample_rate,
            target_sample_rate=target_sample_rate,
            waveform_length=waveform_length,
            normalize=normalize,
            scale_range=scale_range,
            scale_point_range=scale_point_range,
            time_stretch_range=time_stretch_range,
            pitch_shift_range=pitch_shift_range,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length
        )
        augmentation_param_list.append((
            augmentation_params, waves
        ))

    return augmentation_param_list

def _make_param_set_cartesian_product(
        data : tp.Dict[str, torch.Tensor],
        metadata : tp.List[tp.Dict[str, object]],
        seed : int,
        n_augmentations : int,

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
):

    # make basic parameter set
    augmentation_param_list_seq = _make_param_set_sequential(
        data=data,
        metadata=metadata,
        seed=seed,
        n_augmentations=n_augmentations,
        source_sample_rate=source_sample_rate,
        target_sample_rate=target_sample_rate,
        waveform_length=waveform_length,
        normalize=normalize,
        scale_range=scale_range,
        scale_point_range=scale_point_range,
        time_stretch_range=time_stretch_range,
        pitch_shift_range=pitch_shift_range,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )

    # make channel to aug_channel index map
    track_augchannel = dict()
    channel_to_augchannel_map = []
    for t in metadata.tracks:
        if len(channel_to_augchannel_map) == 0:
            channel_to_augchannel_map.append(0)
        elif t and t in track_augchannel:
            channel_to_augchannel_map.append(track_augchannel[t])
        else:
            channel_to_augchannel_map.append(channel_to_augchannel_map[-1]+1)
        # handle track
        if t and t not in track_augchannel:
            track_augchannel[t] = channel_to_augchannel_map[-1]

    augchannel_augment_idx_list = [
        list(range(n_augmentations))
        for _ in range(channel_to_augchannel_map[-1]+1)
    ]

    # select prameter set by cartesian product
    common_params = {
        'normalize': normalize,
        'source_sample_rate': source_sample_rate,
        'target_sample_rate': target_sample_rate,
        'waveform_length': waveform_length,
        'n_fft': n_fft,
        'hop_length': hop_length,
        'win_length': win_length,
    }

    augmentation_param_list = []
    for augchannel_augment_idx in product(*augchannel_augment_idx_list):
        offsets = []
        time_stretch_rates = []
        pitch_shift_rates = []
        scale_amount_list = []
        scale_fraction_list = []
        wave_list = []
        for c_i in range(len(metadata.tracks)):
            ac_i = channel_to_augchannel_map[c_i]
            a_i = augchannel_augment_idx[ac_i]
            offsets.append(
                augmentation_param_list_seq[a_i][0]['offsets'][c_i]
            )
            time_stretch_rates.append(
                augmentation_param_list_seq[a_i][0]['time_stretch_rates'][c_i]
            )
            pitch_shift_rates.append(
                augmentation_param_list_seq[a_i][0]['pitch_shift_rates'][c_i]
            )
            scale_amount_list.append(
                augmentation_param_list_seq[a_i][0]['scale_amount_list'][c_i]
            )
            scale_fraction_list.append(
                augmentation_param_list_seq[a_i][0]['scale_fraction_list'][c_i]
            )
            wave_list.append(
                augmentation_param_list_seq[a_i][1][c_i]
            )

        individual_params = {
            'offsets': offsets,
            'time_stretch_rates': time_stretch_rates,
            'pitch_shift_rates': pitch_shift_rates,
            'scale_amount_list': scale_amount_list,
            'scale_fraction_list': scale_fraction_list,
        }
        augment_param = dict(**common_params, **individual_params)
        waves = torch.stack(wave_list, dim=0)
        augmentation_param_list.append((augment_param, waves))

    return augmentation_param_list


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

        tf = build_transform(**common_params, **params).to(wave.device)
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
    result_waves = []
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
        result_waves.append(Crop(waveform_length, offset).to(wave.device)(wave))

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

    return augment_param, torch.stack(result_waves, dim=0)

