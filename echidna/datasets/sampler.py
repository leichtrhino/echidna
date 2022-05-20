
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
import librosa

from . import transforms

from copy import deepcopy
from itertools import accumulate
from bisect import bisect, bisect_left

logger = logging.getLogger(__name__)


# utility functions/classes for echidna dataset

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

        waveform_length = math.ceil(duration * sr)
        tf = transforms.build_transform(orig_sr,
                                        sr,
                                        waveform_length,
                                        **tf_params)
        return tf_params, tf

class Sampler(torch.utils.data.IterableDataset):
    """
    Dataset sampler class
    """

    def __init__(self,
                 source_list : tp.Dict[str, tp.Dict[str, object]],
                 sr : int,
                 duration : float,
                 source_categories : tp.Union[tp.Set[str], tp.List[str]]=None,
                 num_sources : int=None,
                 category_repetition : bool=False,
                 category_weight : tp.Dict[str, float]=None,
                 splits : tp.List[str]=None,
                 check_track_strictly=True,
                 ) -> None:
        """
        Parameters
        ----------
        source_list : tp.Dict[pathlib.Path, tp.Dict[str, object]]
            - key: wavpath : path to the .wav file
            - midipath : path to the .midi file (optional)
            - category : the category of the source (e.g. vocal, piano)
            - split : the split
            - track : the track identification (if two sources have the same track, the sources are coherent)
        sr : int
        duration : float
        source_categories : tp.List[str] or tp.Set[str]
        num_sources : int
        category_repetition : bool
        splits : tp.List[str]
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
        self.tf_generator = TransformGenerator()

        self.sr = sr
        self.duration = duration
        self._trial_num = 10

    def scale_range(self, scale_range : tp.Tuple[int]):
        self.tf_generator.scale_range = scale_range
        return self

    def scale_point_range(self, scale_point_range : tp.Tuple[int]):
        self.tf_generator.scale_point_range = scale_point_range
        return self

    def pitch_shift_range(self, pitch_shift_range : tp.Tuple[int]):
        self.tf_generator.pitch_shift_range = pitch_shift_range
        return self

    def time_stretch_range(self, time_stretch_range : tp.Tuple[int]):
        self.tf_generator.time_stretch_range = time_stretch_range
        return self

    def normalize(self, normalize : bool):
        self.tf_generator.normalize = normalize
        return self

    def trial_num(self, trial_num):
        self._trial_num = trial_num
        return self

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

    def sample_data_no_retry(self) -> tp.Tuple[tp.Dict[str, torch.Tensor],
                                               tp.Dict[str, object]]:
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
        """

        # loading raw waveform
        sources_dicts = []
        raw_waveforms = []
        activations = []
        track_activation = None
        track_source_count = 0

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
                raise Exception(f'source {source} (track {track_name}) '
                                'does not have activation region')

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
            })
            # reset
            self.source_selector.accept_last_selection()

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
                activation_ = [
                    a for a in activation
                    if len(a[2]) == max_activation
                    and a[0] < source_dict['orig_length'] - num_frames // 2
                ]
                if len(activation_)== 0:
                    source_path = source_dict['path']
                    raise Exception(
                        f'source {source_path} does not have activation')
                start, end, _ = random.choice(activation_)
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

        return {'waveform': waveforms, 'midi': None}, sources_dicts

    def sample_data(self, seed : int=None) -> \
        tp.Tuple[torch.Tensor,
                 tp.List[torch.Tensor],
                 tp.Dict[str, object]]:
        """
        returns
        -------
        torch.Tensor
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
        """

        random.seed(seed)
        for trial_i in range(1, self._trial_num + 2):
            try:
                return self.sample_data_no_retry()
            except Exception as e:
                logger.warning(f'failed on trial {trial_i}/{self._trial_num} : {e}')
        logger.error(f'failed on trial {trial_i}/{self._trial_num} : {e}')
        raise Exception(f'sample_data exceeds maximum trials ({self._trial_num})')


    def _freeze_one(args):
        self, out_dir, out_path_suffix, seed = args
        out_path = os.path.join(out_dir, out_path_suffix)
        data, metadata = self.sample_data(seed)
        for m in metadata:
            m['path'] = str(m['path'])
        torch.save(data, out_path)
        return out_path_suffix, metadata

    def freeze(self,
               out_dir : tp.Union[str, pathlib.Path],
               num_samples : int,
               num_process : int=None) -> None:
        logger.info('Start freezing dataset of %d samples to %s',
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
        seed = lambda: random.randint(0, 2**32-1)
        for out_path_suffix, metadata in map_fn(
                Sampler._freeze_one,
                ((self, out_dir, p, seed()) for p in out_paths)):

            source_metadata[out_path_suffix] = metadata

            sample_count += 1
            logger.info('file wrote (%d / %d)', sample_count, num_samples)

        if not num_process is None:
            pass
        else:
            pool.close()

        with open(os.path.join(out_dir, 'source_metadata.json'), 'w') as fp:
            json.dump(source_metadata, fp)

        logger.info('Finish freezing dataset to %s', out_dir)


    def __iter__(self) -> tp.Iterator[
            tp.Tuple[tp.Dict[str, torch.Tensor], tp.Dict[str, object]]]:
        """
        returns
        -------

        Iterator[tp.Tuple[torch.Tensor, tp.List, tp.Dict['str', object]]
        """
        while True:
            yield self.sample_data()

class FrozenSamples(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir : tp.Union[pathlib.Path, str]) -> None:
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

        self.root_dir = root_dir
        self.source_metadata = source_metadata

    def __len__(self) -> int:
        return len(self.source_metadata)

    def __getitem__(self, idx : int) -> \
        tp.Tuple[tp.Dict[str, torch.Tensor], tp.Dict[str, object]]:
        path, source_metadata = self.source_metadata[idx]
        data = torch.load(path)
        return data, source_metadata

def collate_samples(in_list):
    return (
        {
            'waveform': torch.stack([d[0]['waveform'] for d in in_list]),
            'midi': None,
        },
        [d[1] for d in in_list], # for metadata
    )

