
import typing as tp
import os
import json
import torch

from .transforms import build_transform
from .samples import Sample
from .augmentations import Augmentation
from .mixtures import Mixture


class Dataset(torch.utils.data.Dataset):
    """
    Combine samples, mixtures, augmentations

    """
    def __init__(self,
                 samples_metadata_path,
                 augmentations_metadata_path,
                 mixtures_metadata_path):

        self.samples_metadata_path = samples_metadata_path
        self.augmentations_metadata_path = augmentations_metadata_path
        self.mixtures_metadata_path = mixtures_metadata_path

        with open(samples_metadata_path, 'r') as fp:
            self.samples = Sample.from_list(json.load(fp))

        with open(augmentations_metadata_path, 'r') as fp:
            self.augmentations = sorted(Augmentation.from_list(json.load(fp)),
                                        key=lambda e: (e.augmentation_index,
                                                       e.sample_index))

        with open(mixtures_metadata_path, 'r') as fp:
            self.mixtures = sorted(Mixture.from_list(json.load(fp)),
                                   key=lambda e: (e.mixture_index,
                                                  e.sample_index))

        self.indices = []
        for ai, a in enumerate(self.augmentations):
            for mi, m in enumerate(self.mixtures):
                if a.sample_index != m.sample_index:
                    continue
                for ni, n in enumerate(m.mixture_indices):
                    self.indices.append((a.sample_index, ai, mi, ni))

    def to_dict(self):
        return {
            'samplesPath': self.samples_metadata_path,
            'augmentationsPath': self.augmentations_metadata_path,
            'mixturesPath': self.mixtures_metadata_path,
        }

    @classmethod
    def from_dict(cls, d : dict):
        return cls(
            samples_metadata_path=d['samplesPath'],
            augmentations_metadata_path=d['augmentationsPath'],
            mixtures_metadata_path=d['mixturesPath'],
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_index, augmentation_index, mixture_index, submix_index \
            = self.indices[idx]

        # load sample
        sample_metadata = self.samples[sample_index]
        data = torch.load(
            os.path.join(
                os.path.dirname(self.samples_metadata_path),
                sample_metadata.path
            )
        )

        # augmentation
        augmentation_metadata = self.augmentations[augmentation_index]
        augmented_data = augment(
            data,
            sample_metadata,
            normalize=augmentation_metadata.normalize,

            source_sample_rate=augmentation_metadata.source_sample_rate,
            target_sample_rate=augmentation_metadata.target_sample_rate,
            time_stretch_rates=augmentation_metadata.time_stretch_rates,
            pitch_shift_rates=augmentation_metadata.pitch_shift_rates,
            scale_amount_list=augmentation_metadata.scale_amount_list,
            scale_fraction_list=augmentation_metadata.scale_fraction_list,
            offsets=augmentation_metadata.offsets,
            waveform_length=augmentation_metadata.waveform_length,

            n_fft=augmentation_metadata.n_fft,
            hop_length=augmentation_metadata.hop_length,
            win_length=augmentation_metadata.win_length,
        )

        # mixture
        mixture_metadata = self.mixtures[mixture_index]
        mixture_indices = mixture_metadata.mixture_indices[submix_index]
        mixed_data = mix(data, mixture_indices)

        return mixed_data, \
            {
                'index': idx,
                'sample': sample_metadata,
                'augmentation': augmentation_metadata,
                'mixture': mixture_metadata
            }


def augment(data : tp.Dict[str, torch.Tensor],
            metadata : tp.List[tp.Dict[str, object]],

            normalize : bool,
            source_sample_rate : int,
            target_sample_rate : int,
            time_stretch_rates : tp.List[float],
            pitch_shift_rates : tp.List[float],
            scale_amount_list : tp.List[tp.List[float]],
            scale_fraction_list : tp.List[tp.List[float]],
            offsets : tp.List[int],
            waveform_length : int,

            n_fft : int=2048,
            hop_length : int=512,
            win_length : int=2048,

            ) -> tp.Tuple[
                tp.Dict[str, torch.Tensor],
                tp.List[tp.Dict[str, object]]
            ]:

    num_channels = data['waves'].shape[0]

    assert len(time_stretch_rates) == num_channels
    assert len(pitch_shift_rates) == num_channels
    assert len(scale_amount_list) == num_channels
    assert len(scale_fraction_list) == num_channels
    assert len(offsets) == num_channels

    wave_list = []
    for i in range(num_channels):
        raw_wave = data['waves'][i]

        time_stretch_rate = time_stretch_rates[i]
        pitch_shift_rate = pitch_shift_rates[i]
        scale_amounts = scale_amount_list[i]
        scale_fractions = scale_fraction_list[i]
        offset = offsets[i]

        transform = build_transform(
            normalize=normalize,
            source_sample_rate=source_sample_rate,
            target_sample_rate=target_sample_rate,
            time_stretch_rate=time_stretch_rate,
            pitch_shift_rate=pitch_shift_rate,
            scale_amounts=scale_amounts,
            scale_fractions=scale_fractions,
            offset=offset,
            waveform_length=waveform_length,

            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

        wave_list.append(transform(raw_wave))

    return {
        'waves': torch.stack(wave_list, dim=0),
        'sheets': None,
    }


def mix(data : tp.Dict[str, torch.Tensor],
        out_indices : tp.List[tp.List[int]],
        ) -> tp.Tuple[
            tp.Dict[str, torch.Tensor],
            tp.List[tp.List[tp.Dict[str, object]]],
        ]:
    waves = torch.stack([
        torch.sum(data['waves'][tuple(i), :], dim=0) if len(i) > 0
        else torch.zeros_like(data['waves'][0])
        for i in out_indices
    ])
    sheets = None

    return {
        'waves': waves,
        'sheets': sheets,
    }


