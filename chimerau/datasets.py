
import os
import typing as tp
import math
import pathlib
import json
import random
import itertools
import logging
import traceback
import multiprocessing

import torch
import torchaudio
import resampy
import numpy
import librosa
import matplotlib.pyplot as plt

from . import transforms

from copy import deepcopy
from itertools import accumulate
from bisect import bisect, bisect_left

def plot_partition(x : numpy.ndarray,
                   partitions : tp.List[tp.Tuple[object]]) -> plt.Figure:
    """
    Parameters
    ----------
    x : numpy.Array
        an Array shaped (C, L) where C is the number of sources and L is waveform length
    partitions :
        list of partitions
    """

    pos_dict = {tuple(range(x.shape[0])): (0, 0)}
    for p0, p1 in partitions:
        pall = tuple(sorted(p0+p1))
        row, col = pos_dict[pall]
        pos_dict[tuple(p0)] = (row, col+1)
        pos_dict[tuple(p1)] = (row+len(p0), col+1)

    nrows = max(map(lambda k: k[0], pos_dict.values())) + 1
    ncols = max(map(lambda k: k[1], pos_dict.values())) + 1

    # build spectrograms
    S = numpy.abs(numpy.stack([librosa.stft(y) for y in x], axis=0))
    S_max = numpy.max(numpy.sum(S, axis=0))
    fig, ax = plt.subplots(nrows, ncols)

    for p0, p1, _ in partitions:
        S_p0 = numpy.sum(S[p0], axis=0)
        S_p1 = numpy.sum(S[p1], axis=0)
        S_diff = S_p0 - S_p1
        irow, icol = pos_dict[tuple(sorted(p0+p1))]
        ax[irow, icol].imshow(S_diff,
                              cmap='PiYG',
                              origin='lower',
                              aspect='auto',
                              vmin=-S_max,
                              vmax=S_max)
        ax[irow, icol].set_title(str(tuple(sorted(p0+p1))),
                                 fontsize='x-small',
                                 loc='left')

        if len(p0) == 1:
            irow, icol = pos_dict[tuple(p0)]
            ax[irow, icol].imshow(S_p0,
                                  cmap='PiYG',
                                  origin='lower',
                                  aspect='auto',
                                  vmin=-S_max,
                                  vmax=S_max)
            ax[irow, icol].set_title(str(tuple(p0)),
                                     fontsize='x-small',
                                     loc='left')

        if len(p1) == 1:
            irow, icol = pos_dict[tuple(p1)]
            ax[irow, icol].imshow(-S_p1,
                                  cmap='PiYG',
                                  origin='lower',
                                  aspect='auto',
                                  vmin=-S_max,
                                  vmax=S_max)
            ax[irow, icol].set_title(str(tuple(p1)),
                                     fontsize='x-small',
                                     loc='left')

    for irow, icol in itertools.product(range(nrows), range(ncols)):
        ax[irow, icol].axis('off')

    return fig

def find_partition(x : torch.Tensor,
                   source_dict : tp.Dict[str, tp.Dict[str, object]],
                   n_fft : int=4096,
                   hop_length : int=1024) -> tp.List[tp.Tuple[object]]:
    """
    Parameters
    ----------
    x : torch.Tensor
        a tensor shaped (C, L), where C is the number of sources and L is waveform length
    n_fft : int
    hop_length : int
    Returns
    -------
    list
        a list contains the partition which is suitable for minimum entropy PIT
    """

    win = torch.hann_window(n_fft, device=x.device)
    S = torch.stft(x, n_fft, hop_length, window=win, return_complex=True).abs()

    in_partition = [list(range(x.shape[0]))]
    out_partitions = []
    while in_partition:
        p = in_partition.pop()
        S_all = torch.sum(S[p], dim=0)
        c_min = None
        e_min = math.inf

        for i in range(1, len(p)):
            for c in map(lambda c: list(c), itertools.combinations(p, i)):
                # construct a mixture of selected sources
                S_c = torch.sum(S[c], dim=0)
                weight = S_c / torch.sum(S_c).clamp(min=1e-6)
                # find entropy of this separation
                prob = S_c / S_all.clamp(min=1e-6)
                e = torch.sum(
                    (-prob*torch.log2(prob.clamp(min=1e-6))
                    -(1-prob)*torch.log2((1-prob).clamp(min=1e-6)))
                    *weight
                ).item()
                if e < e_min:
                    e_min = e
                    c_min = c

        not_c = [d for d in p if d not in c_min]
        cs = sorted(
            [c_min, not_c],
            key=lambda c: torch.sum(S[c]),
            reverse=True
        )
        out_partitions.append(cs)
        for c in cs:
            if len(c) > 1:
                in_partition.insert(0, c)

    return out_partitions

def partition_voice_or_not(x : torch.Tensor,
                           source_dict : tp.Dict[str, tp.Dict[str, object]],
                           voice_categories : tp.Set[str]
                           ) -> tp.List[tp.Tuple[object]]:
    """
    Parameters
    ----------
    x : torch.Tensor
        a tensor shaped (C, L), where C is the number of sources and L is waveform length
    source_dict : tp.Dict[str, tp.Dict[str, object]]
    voice_categories : tp.Set[str]
    Returns
    -------
    list
        a list contains the partitions of voice and non-voice mixes
    """
    logger = logging.getLogger(__name__)

    vocal = [
        i for i in range(x.shape[0])
        if source_dict[i].get('category') in voice_categories
    ]
    nonvocal = [
        i for i in range(x.shape[0])
        if source_dict[i].get('category') not in voice_categories
    ]
    if len(vocal) == 0:
        logger.warning(f'sources '
                       f'{[s.get("path", None) for s in source_dict]} does not '
                       f'contain voice (category_set={voice_categories})')
        return []
    if len(vocal) == len(source_dict):
        logger.warning(f'sources '
                       f'{[s.get("path", None) for s in source_dict]} are all '
                       f'voice (category_set={voice_categories})')
        return []

    vocal_combinations = itertools.chain.from_iterable(
        itertools.combinations(vocal, i)
        for i in range(1, len(vocal)+1)
    )
    nonvocal_combinations = itertools.chain.from_iterable(
        itertools.combinations(nonvocal, i)
        for i in range(1, len(nonvocal)+1)
    )

    out_partitions = []
    for v, nv in itertools.product(vocal_combinations, nonvocal_combinations):
        out_partitions.append((v, nv))

    return out_partitions

# utility functions/classes for MEPIT

def get_num_sources_and_category_repetition(
        source_list : tp.Dict[pathlib.Path, tp.Dict[str, object]],
        source_categories : tp.Union[tp.Set[str], tp.List[str]]=None,
        num_sources : int=None,
        category_repetition : bool=False,
) -> int:
    if type(source_categories) == list:
        if not source_categories:
            raise ValueError(
                'source_categories must contain at least one element')
        if num_sources:
            raise ValueError(
                'num_sources must not be given if source_categories is fed')
        num_sources = len(source_categories)
        category_repetition = num_sources > len(set(source_categories))
    elif type(source_categories) == set:
        if not source_categories:
            raise ValueError(
                'source_categories must contain at least one element')
        if num_sources is None:
            num_sources = len(source_categories)
    elif type(source_categories) == type(None):
        if not num_sources:
            raise ValueError(
                'num_sources must be given if source_categories is None')
    return num_sources, category_repetition

def validate_source_list(
        source_list : tp.Dict[pathlib.Path, tp.Dict[str, object]],
        source_categories : tp.Union[tp.Set[str], tp.List[str]]=None,
        num_sources : int=None,
        category_repetition : bool=False,
        category_weight : tp.Dict[str, float]=None,
        splits : tp.List[str]=None,
        check_track_strictly : bool=True
) -> bool:
    """
    Parameters
    ----------
    source_list : tp.Dict[Path, tp.Dict[str, object]]
        the keys of the dict are path to the .wav files, the value contains
        followings :
        - category : the category of the source (e.g. vocal, piano)
        - split : the split
        - track : the track identification (if two sources have the same track,
                  the sources are coherent)
    source_categories : tp.List[str] or tp.Set[str]
    num_sources : int
    category_repetition : bool
    category_weight : tp.Dict[str, float]
    splits : tp.List[str]
    check_track_strictly : bool
    """
    logger = logging.getLogger(__name__)

    num_sources, category_repetition =\
        get_num_sources_and_category_repetition(
            source_list=source_list,
            source_categories=source_categories,
            num_sources=num_sources,
            category_repetition=category_repetition,
        )

    # 候補となるパスを生成する
    # 条件項目:
    #   source_categories: Listの場合、その並び順でソースを生成する
    #                      Setの場合、並びは考慮せず組み合わせのみで生成する
    #                      Noneの場合、全カテゴリが選択肢となる
    # モード系項目:
    # category_repetition: Trueの場合、同一カテゴリの繰り返しが許可される
    #                      Falseの場合、繰り返しは許可されない
    source_list = deepcopy(source_list)

    for p, d in list(source_list.items()):
        category = d.get('category', None)
        split = d.get('split', None)
        if type(source_categories) != type(None) \
           and category not in source_categories:
            del source_list[p]
            continue
        if type(splits) != type(None) \
           and split not in splits:
            del source_list[p]
            continue

    # build track_category_dict : category -> path
    track_category_dict = dict()
    for p, d in source_list.items():
        category = d.get('category', None)
        track = d.get('track', None)
        if (track, category) not in track_category_dict:
            track_category_dict[track, category] = set()
        track_category_dict[track, category].add(p)

    # validate track-category dict
    none_track_count = dict()
    for (t, c), ps in track_category_dict.items():
        if t is None:
            none_track_count[c] = none_track_count.get(c, 0) + len(ps)
    if not category_repetition:
        for k in none_track_count:
            none_track_count[k] = 1

    track_set = set(t for t, c in track_category_dict if t is not None)
    for t in track_set:
        track_count = deepcopy(none_track_count)
        for (u, c), ps in track_category_dict.items():
            if u == t:
                track_count[c] = track_count.get(c, 0) + len(ps)
        if not category_repetition:
            for k in track_count:
                track_count[k] = 1

        if sum(track_count.values()) < num_sources:
            if check_track_strictly:
                raise ValueError(f'track "{t}" contains few sources')
            else:
                logger.warning(f'track "{t}" contains few sources')
                # remove track from source_list, track_category_dict
                for t_, c in list(track_category_dict):
                    if t == t_:
                        del track_category_dict[t, c]
                for sk, sv in list(source_list.items()):
                    if sv.get('track', None) == t:
                        del source_list[sk]

        if type(source_categories) == set and category_repetition:
            for c in source_categories:
                if track_count.get(c, 0) > 0:
                    continue
                if check_track_strictly:
                    raise ValueError(
                        f'track "{t}" does not contain category {c}')
                else:
                    logger.warning(
                        f'track "{t}" does not contain category {c}')
                    # remove track from source_list and track_category_dict
                    for t_, c_ in list(track_category_dict):
                        if t == t_:
                            del track_category_dict[t_, c_]
                    for sk, sv in list(source_list.items()):
                        if sv.get('track', None) == t:
                            del source_list[sk]

        elif type(source_categories) == set and not category_repetition:
            pass
        elif type(source_categories) == list:
            source_cat_count = dict()
            for c in source_categories:
                source_cat_count[c] = source_cat_count.get(c, 0) + 1
            for c, count in source_cat_count.items():
                if track_count.get(c, 0) >= count:
                    continue
                if check_track_strictly:
                    raise ValueError(
                        f'track "{t}" contains few sources of category {c}')
                else:
                    logger.warning(
                        f'track "{t}" contains few sources of category {c}')
                    # remove track from source_list and track_category_dict
                    for t_, c_ in list(track_category_dict):
                        if t == t_:
                            del track_category_dict[t_, c_]
                    for sk, sv in list(source_list.items()):
                        if sv.get('track', None) == t:
                            del source_list[sk]
    # end for track_set

    if not track_set:
        if sum(none_track_count.values()) < num_sources:
            raise ValueError('source_list contains few sources')

        if type(source_categories) == set and category_repetition:
            for c in source_categories:
                if none_track_count.get(c, 0) == 0:
                    raise ValueError(
                        'source_list does not contain category {c}')
        elif type(source_categories) == set and not category_repetition:
            pass
        elif type(source_categories) == list:
            source_cat_count = dict()
            for c in source_categories:
                source_cat_count[c] = source_cat_count.get(c, 0) + 1
            for c, count in source_cat_count.items():
                if none_track_count.get(c, 0) < count:
                    raise ValueError(
                        'track "{t}" contains few sources of category {c}')

    return source_list, track_category_dict

def merge_activation(base_list : tp.List[tp.Tuple[int, int, tp.List[str]]],
                     x : torch.Tensor,
                     tag : str,
                     top_db : float=60,
                     frame_length : int=2048,
                     hop_length : int=512) -> tp.List[tp.Tuple[int, int, str]]:
    # initial state of activation_list is [(0, length, [])]
    # calculate activation from silence
    activations = librosa.effects.split(x.numpy(),
                                        top_db=top_db,
                                        frame_length=frame_length,
                                        hop_length=hop_length)

    for a_f, a_t in activations:
        # find leftmost index
        i = bisect([b[0] for b in base_list], a_f) - 1

        # divide current section into two segments
        if i < len(base_list) and base_list[i][0] < a_f:
            b_s, b_t, b_tag = base_list[i]
            base_list[i] = (b_s, a_f, b_tag)
            base_list.insert(i+1, (a_f, b_t, b_tag[:]))
            i += 1

        # while adding tag to list, find rightmost index
        while i < len(base_list) and base_list[i][1] <= a_t:
            base_list[i][2].append(tag)
            i += 1

        # divide current section into two segments
        if i < len(base_list) and base_list[i][0] < a_t:
            b_s, b_t, b_tag = base_list[i]
            base_list[i] = (a_t, b_t, b_tag)
            base_list.insert(i, (b_s, a_t, b_tag+[tag]))


class SourceSelector(object):
    def __init__(self,
                 source_list : tp.Dict[pathlib.Path, tp.Dict[str, object]],
                 source_categories : tp.Union[tp.Set[str], tp.List[str]]=None,
                 num_sources : int=None,
                 category_repetition : bool=False,
                 category_weight : tp.Dict[str, float]=None,
                 splits : tp.List[str]=None,
                 check_track_strictly : bool=True) -> None:

        self.num_sources, self.category_repetition =\
            get_num_sources_and_category_repetition(
                source_list=source_list,
                source_categories=source_categories,
                num_sources=num_sources,
                category_repetition=category_repetition,
            )

        self.source_list, self.track_category_dict_orig =\
            validate_source_list(source_list=source_list,
                                 source_categories=source_categories,
                                 num_sources=num_sources,
                                 category_repetition=category_repetition,
                                 category_weight=category_weight,
                                 splits=splits,
                                 check_track_strictly=check_track_strictly)

        self.source_categories = deepcopy(source_categories)
        self.category_weight = deepcopy(category_weight)
        self.splits = deepcopy(splits)

        self.reset()

    def has_next(self):
        return self.source_i < self.num_sources

    def reset(self):
        self.source_i = 0
        self.last_source = None
        self.track_category_dict = deepcopy(self.track_category_dict_orig)

    def select_source(self):
        # select category first
        if type(self.source_categories) == list:
            category = self.source_categories[self.source_i]
        elif self.source_categories is None or type(self.source_categories) == set:
            category_list = list(set(
                c for t, c in self.track_category_dict
            ))
            if self.category_weight:
                category_weight = [
                    max(self.category_weight.get(c, 0), 0)
                    for c in category_list
                ]
                if all(w == 0 for w in category_weight):
                    category_weight = [1] * len(category_weight)
                category_weight = list(accumulate(category_weight))
                rvalue = random.uniform(0, category_weight[-1])
                category = category_list[bisect_left(
                    category_weight, rvalue)]
            else:
                category = random.choice(category_list)

        # take sample of selected category
        source_list = sum([
            list(ps) for (t, c), ps in self.track_category_dict.items()
            if c == category
        ], [])
        source = random.choice(source_list)

        self.last_source = source
        return source

    def accept_last_selection(self):
        # remove last_category and last_source from source, category dict
        if self.last_source is None:
            raise ValueError('no source selected')

        track = self.source_list[self.last_source].get('track', None)
        category = self.source_list[self.last_source].get('category', None)
        self.track_category_dict[track, category].remove(self.last_source)
        if len(self.track_category_dict[track, category]) == 0:
            del self.track_category_dict[track, category]

        # remove from track_category_dict other than the track or category
        for t, c in list(self.track_category_dict):
            if track is not None and t is not None and t != track:
                del self.track_category_dict[(t, c)]
                continue
            if not self.category_repetition and \
               category is not None and c is not None and c == category:
                del self.track_category_dict[(t, c)]
                continue

        # increment source_i
        self.source_i += 1
        self.last_source = None

class TransformGenerator(object):
    def __init__(self,
                 scale_range : tp.Tuple[float, float]=(1.0, 1.0),
                 scale_point_range : tp.Tuple[int, int]=(2, 2),
                 pitch_shift_range : tp.Tuple[float, float]=(1.0, 1.0),
                 time_stretch_range : tp.Tuple[float, float]=(1.0, 1.0),
                 normalize : bool=False,
                 ) -> None:
        self.scale_range = scale_range
        self.scale_point_range = scale_point_range
        self.pitch_shift_range = pitch_shift_range
        self.time_stretch_range = time_stretch_range
        self.normalize = normalize

    def generate_transform_with_params(self,
                                       orig_sr : int,
                                       sr : int,
                                       duration : float,
                                       **tf_params):
        unif = random.uniform
        if len(tf_params) == 0:
            scale_point = random.randint(*self.scale_point_range)
            tf_params = {
                'time_stretch_rate': unif(*self.time_stretch_range),
                'pitch_shift_rate': unif(*self.pitch_shift_range),
                'scales': [unif(*self.scale_range) for _ in range(scale_point)],
                'scale_duration': [unif(0, 1) for _ in range(scale_point-2)],
                'normalize': self.normalize,
            }

        '''
        num_frames = math.ceil(
            orig_sr * duration * tf_params['time_stretch_rate'])
        offset = max(0,
                     int(random.uniform(0, metadata.num_frames - num_frames)))
        '''

        waveform_length = math.ceil(duration * sr)
        tf = transforms.build_transform(orig_sr,
                                        sr,
                                        waveform_length,
                                        **tf_params)
        return tf_params, tf

class MEPIT(torch.utils.data.IterableDataset):
    """
    Dataset for ME-PIT
    """

    def __init__(self,
                 source_list : tp.Dict[pathlib.Path, tp.Dict[str, object]],
                 sr : int,
                 duration : float,
                 source_categories : tp.Union[tp.Set[str], tp.List[str]]=None,
                 num_sources : int=None,
                 category_repetition : bool=False,
                 category_weight : tp.Dict[str, float]=None,
                 splits : tp.List[str]=None,
                 scale_range : tp.Tuple[float, float]=(1.0, 1.0),
                 scale_point_range : tp.Tuple[float, float]=(2, 2),
                 pitch_shift_range : tp.Tuple[float, float]=(1.0, 1.0),
                 time_stretch_range : tp.Tuple[float, float]=(1.0, 1.0),
                 normalize : bool=False,
                 top_db : float=60,
                 check_track_strictly=True,
                 partition_algorithm : callable=None,
                 partition_arguments : tp.Dict[str, object]=None,
                 ) -> None:
        """
        Parameters
        ----------
        source_list : tp.List[tp.Dict[Path, object]]
            the keys of the dict are path to the .wav files, the value contains followings :
            - category : the category of the source (e.g. vocal, piano)
            - split : the split
            - track : the track identification (if two sources have the same track, the sources are coherent)
        sr : int
        duration : float
        source_categories : tp.List[str] or tp.Set[str]
        num_sources : int
        category_repetition : bool
        splits : tp.List[str]
        scale_range : tp.Tuple[float, float]
        scale_point_range : tp.Tuple[float, float]
        pitch_shift_range : tp.Tuple[float, float]
        time_stretch_range : tp.Tuple[float, float]
        normalize : bool
        """

        self.source_list = deepcopy(source_list)
        self.source_selector = SourceSelector(
            source_list=source_list,
            source_categories=source_categories,
            num_sources=num_sources,
            category_repetition=category_repetition,
            category_weight=category_weight,
            splits=splits,
            check_track_strictly=check_track_strictly,
        )
        self.tf_generator = TransformGenerator(
            scale_range=scale_range,
            scale_point_range=scale_point_range,
            pitch_shift_range=pitch_shift_range,
            time_stretch_range=time_stretch_range,
            normalize=normalize,
        )

        self.sr = sr
        self.duration = duration
        self.partition_algorithm = partition_algorithm
        self.partition_arguments = partition_arguments

    def sample_source(self) -> tp.List[str]:
        """
        @deprecated remains the method for backward compatibility
        returns
        -------
        list of selected keys from source_list
        """

        sources = []
        self.source_selector.reset()
        while self.source_selector.has_next():
            source = self.source_selector.select_source()
            sources.append(source)
            self.source_selector.accept_last_selection()

        return sources

    def _sample_data(self, max_trial_num=3) -> tp.Dict[str, object]:
        """
        returns
        -------
        dict
        'data': sampled tensors, shape is (c, l)
                where c is the channel and l is length of the samples
        'sources': the list of source information dictionary
            each dictionary contains followings:
                'category': the category of the source
                'track': the track of the source
                'path': the source file which the sample came from
                'start': start time (in second)
                'end': end time (in second)
                'tf_params': parameters of transforms
        'partitions' : the list of partitions which satisfies minimum entropy pit
            each partition is a tuple contains (source)
        """
        logger = logging.getLogger(__name__)

        # loading raw waveform
        sources_dicts = []
        raw_waveforms = []
        activations = []
        track_activation = None
        track_source_count = 0
        trial_num = 0

        self.source_selector.reset()
        while self.source_selector.has_next():
            # load
            source = self.source_selector.select_source()
            x, orig_sr = torchaudio.load(source)
            x = x.mean(dim=0)

            # evaluate source to append (dry-run)
            if self.source_list[source].get('track'):
                # merge activation window and check if they have overlaps
                if track_activation is None:
                    dummy_activation = [(0, x.shape[-1], [])]
                else:
                    dummy_activation = deepcopy(track_activation)
                source_count = track_source_count
            else:
                dummy_activation = [(0, x.shape[-1], [])]
                source_count = 0

            # evaluation
            merge_activation(dummy_activation, x, tag=source)
            if max([len(a[2]) for a in dummy_activation]) \
               < source_count + 1:
                track_name = self.source_list[source].get('track', None)
                logger.warning(f'source {source} (track {track_name}) '
                               'does not have activation region '
                               f'(trial={trial_num}/{max_trial_num})')
                trial_num += 1
                if trial_num >= max_trial_num:
                    raise Exception(f'source {source} (track {track_name}) '
                                    'does not have activation region '
                                    f'(trial={trial_num}/{max_trial_num})')
                continue

            # re-calculate activation and add it
            if self.source_list[source].get('track'):
                # merge activation window and check if they have overlaps
                if track_activation is None:
                    track_activation = [(0, x.shape[-1], [])]
                activation_ = track_activation
                track_source_count += 1
            else:
                activation_ = [(0, x.shape[-1], [])]
            merge_activation(activation_, x, tag=source)
            activations.append(activation_)

            # adding it to waveform
            raw_waveforms.append(x)
            sources_dicts.append({
                'path': source,
                'category': self.source_list[source].get('category', None),
                'track': self.source_list[source].get('track', None),
                'orig_sr': orig_sr,
                'orig_length': x.shape[-1]
                #'sr': self.sr,
                #'length': x.shape[-1],
                #'start': offset / orig_sr,
                #'end': (offset + num_frames) / orig_sr,
                #'tf_params': tf_params,
            })
            # reset
            self.source_selector.accept_last_selection()
            trial_num = 0

        # transforming
        track_tf_params = None
        track_tf = None
        track_num_frames = None
        track_offset = None
        waveforms = []
        for source_dict, raw_waveform, activation in \
            zip(sources_dicts, raw_waveforms, activations):

            # get transform parameter
            if source_dict['track'] and track_tf is not None:
                # set parameter
                tf_params = deepcopy(track_tf_params)
                tf = track_tf
                num_frames = track_num_frames
                offset = track_offset

            else:
                # init parameter
                tf_params, tf =\
                    self.tf_generator.generate_transform_with_params(
                        source_dict['orig_sr'], self.sr, self.duration
                    )
                num_frames = math.ceil(
                    source_dict['orig_sr']
                    * self.duration
                    * tf_params['time_stretch_rate'])

                # get activations
                max_activation = max(len(a[2]) for a in activation)
                # TODO parametrize "2"
                activation_ = [
                    a for a in activation
                    if len(a[2]) == max_activation
                    and a[0] < source_dict['orig_length'] - num_frames // 2
                ]
                if len(activation_)== 0:
                    source_path = source_dict['path']
                    logger.warning(
                        f'source {source_path} does not have activation')
                    raise Exception(
                        f'source {source_path} does not have activation')
                start, end, _ = random.choice(activation_)
                # TODO parametrize "4"
                start = max(0, start - num_frames // 4)
                end = min(source_dict['orig_length'], end + num_frames // 4)
                end = max(start, end - num_frames)
                offset = random.randint(start, end)

            # set transform parameters for same track
            if source_dict.get('track') and track_tf is None:
                track_tf_params = deepcopy(tf_params)
                track_tf = tf
                track_num_frames = num_frames
                track_offset = offset

            # transform and update parameters
            waveform = tf(raw_waveform[offset:offset+num_frames])
            waveforms.append(waveform)
            source_dict.update({
                'sr': self.sr,
                'length': waveform.shape[-1],
                'start': offset / source_dict['orig_sr'],
                'end': (offset + num_frames) / source_dict['orig_sr'],
                'tf_params': tf_params,
            })

        waveforms = torch.stack(waveforms, dim=0)
        if self.partition_algorithm is None:
            partition_func = find_partition
        else:
            partition_func = self.partition_algorithm
        if self.partition_arguments is None:
            partition_arguments = {}
        else:
            partition_arguments = self.partition_arguments

        return waveforms,\
            sources_dicts,\
            partition_func(waveforms, sources_dicts, **partition_arguments)

    def sample_data(self,
                    seed : int=None,
                    sample_trial_num : int=10,
                    track_trial_num : int = 3) -> tp.Dict[str, object]:
        """
        returns
        -------
        dict
        'data': sampled tensors, shape is (c, l) where c is the channel
                and l is length of the samples
        'sources': the list of source information dictionary
            each dictionary contains followings:
                'category': the category of the source
                'track': the track of the source
                'path': the source file which the sample came from
                'start': start time (in second)
                'end': end time (in second)
                'tf_params': parameters of transforms
        'partitions' : the list of partitions which satisfies minimum
                       entropy pit each partition is a tuple contains (source)
        """

        logger = logging.getLogger(__name__)
        random.seed(seed)
        for trial_i in range(1, sample_trial_num + 2):
            try:
                return self._sample_data(track_trial_num)
            except Exception as e:
                logger.warning(f'failed on trial {trial_i} : {e}')
        raise Exception(f'sample_data exceeds maximum trials ({trial_num})')


    def _freeze_one(args):
        out_dir, out_path_suffix, MEPIT, seed, sample_trials, track_trials = args
        out_path = os.path.join(out_dir, out_path_suffix)
        data, metadata, partition = MEPIT.sample_data(seed,
                                                      sample_trials,
                                                      track_trials)
        for m in metadata:
            m['path'] = str(m['path'])
        torch.save(data, out_path)
        return out_path_suffix, metadata, partition

    def freeze(self,
               out_dir : tp.Union[str, pathlib.Path],
               num_samples : int,
               num_process : int=None,
               sample_trials : int=10,
               track_trials : int=3) -> None:
        logger = logging.getLogger(__name__)
        logger.info('Start freezing MEPIT dataset of %d samples to %s',
                    num_samples,
                    out_dir)

        directory_depth = 0
        ns_copy = num_samples - 1
        while ns_copy > 0:
            directory_depth += 1
            ns_copy = ns_copy // 1000

        out_paths = []

        for si in range(num_samples):
            si_digits = ('{:0' + str(directory_depth*3) + 'd}').format(si)
            out_path_suffix = os.path.join(
                *[
                    si_digits[di*3:(di+1)*3]
                    for di in range(0, directory_depth-1)
                ],
                si_digits + '.pth'
            )
            out_path = os.path.join(out_dir, out_path_suffix)

            if not os.path.exists(os.path.dirname(out_path)):
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

            out_paths.append(out_path_suffix)


        if num_process is None:
            map_fn = map
        else:
            pool = multiprocessing.Pool(num_process)
            map_fn = pool.imap_unordered

        source_metadata = dict()
        partitions = dict()

        sample_count = 0
        for out_path_suffix, metadata, partition\
            in map_fn(MEPIT._freeze_one,
                      ((out_dir, p, self,
                        random.randint(0, 2**32-1), sample_trials, track_trials)
                       for p in out_paths)):

            source_metadata[out_path_suffix] = metadata
            partitions[out_path_suffix] = partition

            sample_count += 1
            if sample_count % 1000 == 0:
                logger.info('file wrote (%d / %d)', sample_count, num_samples)

        if not num_process is None:
            pass
        else:
            pool.close()

        with open(os.path.join(out_dir, 'source_metadata.json'), 'w') as fp:
            json.dump(source_metadata, fp)

        with open(os.path.join(out_dir, 'partition_metadata.json'), 'w') as fp:
            json.dump(partitions, fp)

        logger.info('Finish freezing MEPIT dataset to %s', out_dir)


    def __iter__(self) -> tp.Iterator[
            tp.Tuple[torch.Tensor, tp.Dict['str', object], tp.List]]:
        """
        returns
        -------

        Iterator[tp.Tuple[torch.Tensor, tp.List, tp.Dict['str', object]]
        """
        while True:
            yield self.sample_data()


class FrozenMEPIT(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir : tp.Union[pathlib.Path, str],
                 out_partition_metadata : bool=False,
                 out_source_metadata : bool=False) -> None:
        # load source_metadata.json
        sm_path = os.path.join(root_dir, 'source_metadata.json')
        try:
            with open(sm_path, 'r') as fp:
                source_metadata = json.load(fp)
        except OSError as err:
            raise err
        except ValueError as err:
            raise err
        source_metadata = sorted(
            (os.path.join(root_dir, k), v)
            for k, v in source_metadata.items())

        # check all source exist
        lack_files = []
        for s, _ in source_metadata:
            if not os.path.exists(s):
                lack_files.append(s)
        if lack_files:
            raise ValueError(f'paths {lack_files} does not exist')

        # load partition_metadata.json
        pm_path = os.path.join(root_dir, 'partition_metadata.json')
        try:
            with open(pm_path, 'r') as fp:
                partition_metadata = json.load(fp)
        except OSError as err:
            raise err
        except ValueError as err:
            raise err
        partition_metadata = sorted(
            (os.path.join(root_dir, k), v)
            for k, v in partition_metadata.items())

        # check all source exist
        lack_files = []
        for s, _ in partition_metadata:
            if not os.path.exists(s):
                lack_files.append(s)
        if lack_files:
            raise ValueError(f'paths {lack_files} does not exist')

        # check source_metadata and partition_metadata shares same sources
        if [s for s, _ in source_metadata] \
           != [s for s, _ in partition_metadata]:
            raise ValueError(
                'paths of source_metadata and partition_metadata differs')

        self.out_partition_metadata = out_partition_metadata
        self.out_source_metadata = out_source_metadata
        self.source_metadata = source_metadata
        self.partition_metadata = partition_metadata

    def __len__(self) -> int:
        return len(self.source_metadata)

    def __getitem__(self, idx : int) -> \
        tp.Union[torch.Tensor,
                 tp.Tuple[torch.Tensor, tp.Dict],
                 tp.Tuple[torch.Tensor, tp.List],
                 tp.Tuple[torch.Tensor, tp.List, tp.Dict]]:
        path, source_metadata = self.source_metadata[idx]
        _, partition_metadata = self.partition_metadata[idx]
        data = torch.load(path)
        if self.out_partition_metadata and self.out_source_metadata:
            return data, source_metadata, partition_metadata
        elif self.out_partition_metadata and not self.out_source_metadata:
            return data, partition_metadata
        elif not self.out_partition_metadata and self.out_source_metadata:
            return data, source_metadata
        elif not self.out_partition_metadata and not self.out_source_metadata:
            return data
        else:
            return None

    def collate_fn_with_partition_and_source(l):
        return (
            torch.stack([m[0] for m in l], dim=0),
            [m[1] for m in l],
            [m[2] for m in l],
        )

    def collate_fn_with_partition(l):
        return (
            torch.stack([m[0] for m in l], dim=0),
            [m[1] for m in l],
        )

    def collate_fn_with_source(l):
        return (
            torch.stack([m[0] for m in l], dim=0),
            [m[1] for m in l],
        )

    def collate_fn_without_metadata(l):
        return torch.stack([m[0] for m in l], dim=0)


    def get_collate_function(self):
        def collate_function(l):
            if self.out_partition_metadata and self.out_source_metadata:
                return (
                    torch.stack([m[0] for m in l], dim=0),
                    [m[1] for m in l],
                    [m[2] for m in l],
                )
            elif self.out_partition_metadata and not self.out_source_metadata:
                return (
                    torch.stack([m[0] for m in l], dim=0),
                    [m[1] for m in l],
                )
            elif not self.out_partition_metadata and self.out_source_metadata:
                return (
                    torch.stack([m[0] for m in l], dim=0),
                    [m[1] for m in l],
                )
            elif not self.out_partition_metadata and not self.out_source_metadata:
                return torch.stack([m[0] for m in l], dim=0),
            else:
                return None
        return collate_function


class FrozenMEPITMixByPartition(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir : tp.Union[pathlib.Path, str],
                 waveform_length : int=None,
                 random_sample : bool=False) -> None:
        # load source_metadata.json
        sm_path = os.path.join(root_dir, 'source_metadata.json')
        try:
            with open(sm_path, 'r') as fp:
                source_metadata = json.load(fp)
        except OSError as err:
            raise err
        except ValueError as err:
            raise err
        source_metadata = sorted(
            (os.path.join(root_dir, k), v)
            for k, v in source_metadata.items())

        # check all source exist
        lack_files = []
        for s, _ in source_metadata:
            if not os.path.exists(s):
                lack_files.append(s)
        if lack_files:
            raise ValueError(f'paths {lack_files} does not exist')

        # load partition_metadata.json
        pm_path = os.path.join(root_dir, 'partition_metadata.json')
        try:
            with open(pm_path, 'r') as fp:
                partition_metadata = json.load(fp)
        except OSError as err:
            raise err
        except ValueError as err:
            raise err
        partition_metadata = sorted(
            (os.path.join(root_dir, k), v)
            for k, v in partition_metadata.items())

        # check all source exist
        lack_files = []
        for s, _ in partition_metadata:
            if not os.path.exists(s):
                lack_files.append(s)
        if lack_files:
            raise ValueError(f'paths {lack_files} does not exist')

        # check source_metadata and partition_metadata shares same sources
        if [s for s, _ in source_metadata] \
           != [s for s, _ in partition_metadata]:
            raise ValueError(
                'paths of source_metadata and partition_metadata differs')

        self.source_metadata = source_metadata
        self.partition_metadata = partition_metadata
        self.source_indices = []
        for _, p in partition_metadata:
            self.source_indices.append(
                len(p)
                + (0 if len(self.source_indices) == 0 else
                   self.source_indices[-1])
            )
        self.waveform_length = waveform_length
        self.random_sample = random_sample

    def __len__(self) -> int:
        return self.source_indices[-1] if len(self.source_indices) > 0 else 0

    def __getitem__(self, idx : int) -> torch.Tensor:
        s_idx = bisect_left(self.source_indices, idx)
        offset = idx - self.source_indices[s_idx]
        path, source_metadata = self.source_metadata[s_idx]
        _, partition_metadata = self.partition_metadata[s_idx]
        data = torch.load(path)
        stacked = torch.stack(
            [torch.sum(data[list(p), :], dim=0)
             for p in partition_metadata[offset]],
            dim=0
        )
        if self.waveform_length is None \
           or self.waveform_length >= stacked.shape[-1]:
            return stacked

        threshold = 0.1
        signal_abs = torch.min(stacked.abs(), axis=0)[0]
        pos = torch.arange(stacked.shape[-1])[signal_abs > threshold]
        if pos.numel() == 0:
            pos = torch.argmax(signal_abs, dim=-1)
        elif self.random_sample:
            pos = random.choice(pos)
        else:
            pos = pos[0]

        if pos < self.waveform_length // 2:
            left = 0
            right = min(stacked.shape[-1], self.waveform_length)
        elif pos > stacked.shape[-1] - self.waveform_length // 2:
            left = max(0, stacked.shape[-1] - self.waveform_length)
            right = stacked.shape[-1]
        else:
            left = pos - self.waveform_length // 2
            right = left + self.waveform_length

        return stacked[..., left:right]

