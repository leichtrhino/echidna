
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

from .sampler import Sampler, FrozenSamples

from copy import deepcopy
from itertools import product, combinations
from bisect import bisect, bisect_left

logger = logging.getLogger(__name__)

def mix(data : tp.Dict[str, torch.Tensor],
        metadata : tp.List[tp.Dict[str, object]],
        out_indices : tp.List[tp.List[int]],
        ) -> tp.Tuple[
            tp.Dict[str, torch.Tensor],
            tp.List[tp.List[tp.Dict[str, object]]],
        ]:
    waveform = torch.stack([
        torch.sum(data['waveform'][tuple(i), :], dim=0) if len(i) > 0
        else torch.zeros_like(data['waveform'][0])
        for i in out_indices
    ])
    midi = None
    out_metadata = [[metadata[j] for j in i] for i in out_indices]
    return {
        'waveform': waveform,
        'midi': midi,
    }, out_metadata

class MixAlgorithm(object):
    def mix_index(self,
                  data : tp.Dict[str, torch.Tensor],
                  metadata : tp.List[tp.Dict[str, object]]
                  ) -> tp.List[tp.List[tp.List[int]]]:
        """
        # mix_samples, mix_out, mix_in
        """
        raise NotImplementedError()

class CategoryMix(MixAlgorithm):
    def __init__(self,
                 mix_category_list : tp.List[tp.Union[str, tp.List[str]]],
                 output_other : bool=True,
                 check_duplicate : bool=True):
        self.mix_categories = []
        all_categories = set()
        for i, l in enumerate(mix_category_list):
            self.mix_categories.append([])
            if type(l) == str:
                if l in all_categories:
                    logging.warning(f'category {l} at {i} duplicates')
                    if check_duplicate:
                        raise ValueError(f'category {l} at {i} duplicates')
                self.mix_categories[i].append(l)
                all_categories.add(l)
                continue
            for j, c in enumerate(l):
                if c in all_categories:
                    logging.warning(f'category {c} at {i} duplicates')
                    if check_duplicate:
                        raise ValueError(f'category {c} at {i} duplicates')
                self.mix_categories[i].append(c)
                all_categories.add(c)
        self.output_other = output_other

    def mix_index(self,
                  data : tp.Dict[str, torch.Tensor],
                  metadata : tp.List[tp.Dict[str, object]]
                  ) -> tp.List[tp.List[tp.List[int]]]:
        """
        # mix_samples, mix_out, mix_in
        """
        base_index = [[] for _ in range(len(self.mix_categories))]
        if self.output_other:
            base_index.append([])

        for i, c in enumerate(m['category'] for m in metadata):
            is_other = True
            for j, ds in enumerate(self.mix_categories):
                if c in ds:
                    base_index[j].append(i)
                    is_other = False
            if self.output_other and is_other:
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

        return list(product(*mix_index))


class Mixer(torch.utils.data.IterableDataset):
    def __init__(self,
                 sampler : Sampler,
                 algorithm : MixAlgorithm) -> None:
        self.sampler = sampler
        self.algorithm = algorithm

    def __iter__(self):
        sampler_iter = iter(self.sampler)
        while True:
            data, metadata = next(sampler_iter)
            indices = self.algorithm.mix_index(data, metadata)
            for out_indices in indices:
                yield mix(data, metadata, out_indices)

def freeze_mix(sampler : tp.Union[str, pathlib.Path, FrozenSamples],
               algorithm : MixAlgorithm,
               name : str):
    if type(sampler) == str or isinstance(sampler, pathlib.Path):
        sampler = FrozenSamples(root_dir=sampler)
    elif type(sampler) == FrozenSamples:
        pass
    else:
        raise TypeError(f'the type of sampler should be either '
                        f'[str, Path, FrozenSamples] but {type(sampler)} '
                        'is given')

    mix_indices = dict()
    for i in range(len(sampler)):
        data, metadata = sampler[i]
        path, _ = sampler.source_metadata[i]
        path = os.path.basename(path) # NOTE: the path is os.path.join-ed
        mix_indices[path] = algorithm.mix_index(data, metadata)

    mix_root = os.path.join(sampler.root_dir, 'mix')
    if not os.path.exists(mix_root):
        os.makedirs(mix_root)

    with open(os.path.join(mix_root, name), 'w') as fp:
        json.dump(mix_indices, fp)


class FrozenMix(torch.utils.data.Dataset):
    def __init__(self,
                 sampler : tp.Union[str, pathlib.Path, FrozenSamples],
                 name : str) -> None:
        # load index information from name
        if type(sampler) == str or isinstance(sampler, pathlib.Path):
            sampler = FrozenSamples(root_dir=sampler)
        elif type(sampler) == FrozenSamples:
            pass
        else:
            raise TypeError(f'the type of sampler should be either '
                            f'[str, Path, FrozenSamples] but {type(sampler)} '
                            'is given')

        self.sampler = sampler
        mix_path = os.path.join(sampler.root_dir, 'mix', name)
        with open(mix_path, 'r') as fp:
            mix_indices_dict = dict(
                (os.path.join(sampler.root_dir, bn), m)
                for bn, m in json.load(fp).items()
            )

        self.mix_indices = []
        self.mix_indices_size = []
        for p, _ in sampler.source_metadata:
            m = mix_indices_dict.get(p, [])
            self.mix_indices.append(m)
            self.mix_indices_size.append(
                len(m)
                + (0 if len(self.mix_indices_size) == 0 else
                   self.mix_indices_size[-1])
            )

    def __len__(self) -> int:
        return self.mix_indices_size[-1] if len(self.mix_indices_size) > 0 else 0

    def __getitem__(self, idx) -> \
        tp.Tuple[tp.Dict[str, torch.Tensor], tp.List[tp.Dict[str, object]]]:
        s_idx = bisect_left(self.mix_indices_size, idx)
        offset = idx - self.mix_indices_size[s_idx]
        path, metadata = self.sampler.source_metadata[s_idx]
        mix_index = self.mix_indices[s_idx][offset]

        data = torch.load(path)

        return mix(data, metadata, mix_index)

def collate_mix(in_list):
    return (
        {
            'waveform': torch.stack([d[0]['waveform'] for d in in_list]),
            'midi': None,
        },
        [d[1] for d in in_list], # for metadata
    )
