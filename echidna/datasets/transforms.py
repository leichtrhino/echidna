
import typing as tp
import math
import random
import resampy
import torch
import torchaudio

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class Resample(torch.nn.Module):
    """
    Resample
    """

    def __init__(self, orig_freq : int, new_freq : int) -> None:
        """
        Parameters
        ----------
        """

        super(Resample, self).__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------

        Returns
        -------
        """
        return torch.Tensor(resampy.resample(
            x.numpy(), self.orig_freq, self.new_freq, axis=-1))

class TimeStretch(torch.nn.Module):
    """
    TimeStretch
    """

    def __init__(self,
                 time_stretch_rate : float,
                 n_fft : int=2048,
                 hop_length : int=512,
                 win_length : int=2048) -> None:
        """
        Parameters
        ----------
        """

        super(TimeStretch, self).__init__()
        self.stft = lambda x: torch.stft(
            x, n_fft, hop_length=hop_length,
            window=torch.hann_window(win_length),
            return_complex=True,
        )
        self.istft = lambda x: torch.istft(
            x, n_fft, hop_length=hop_length,
            window=torch.hann_window(win_length),
        )
        self.time_stretch = torchaudio.transforms.TimeStretch(
            hop_length=hop_length,
            n_freq=(n_fft//2)+1,
            fixed_rate=time_stretch_rate
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------

        Returns
        -------
        """
        return self.istft(self.time_stretch(self.stft(x)))

class PitchShift(torch.nn.Module):
    """
    PitchShift
    """

    def __init__(self,
                 sampling_rate : int,
                 shift_rate : float,
                 n_fft : int=2048,
                 hop_length : int=512,
                 win_length : int=2048) -> None:
        """
        Parameters
        ----------
        """

        super(PitchShift, self).__init__()
        self.stft = lambda x: torch.stft(
            x, n_fft, hop_length=hop_length,
            window=torch.hann_window(win_length),
            return_complex=True,
        )
        self.istft = lambda x: torch.istft(
            x, n_fft, hop_length=hop_length,
            window=torch.hann_window(win_length),
        )
        self.time_stretch = torchaudio.transforms.TimeStretch(
            hop_length=hop_length,
            n_freq=(n_fft//2)+1,
            fixed_rate=shift_rate
        )
        self.resample = Resample(
            orig_freq=sampling_rate/shift_rate,
            new_freq=sampling_rate
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------

        Returns
        -------
        """

        spec_orig = self.stft(x)
        spec_target = self.time_stretch(spec_orig)
        x_target = self.istft(spec_target)
        return self.resample(x_target)

class TimeStretchAndPitchShift(torch.nn.Module):
    """
    TimeStretchAndPitchShift
    """

    def __init__(self,
                 sampling_rate : int,
                 stretch_rate : float,
                 shift_rate : float,
                 n_fft : int=2048,
                 hop_length : int=512,
                 win_length : int=2048) -> None:
        """
        Parameters
        ----------
        """

        super(TimeStretchAndPitchShift, self).__init__()
        self.stft = lambda x: torch.stft(
            x, n_fft, hop_length=hop_length,
            window=torch.hann_window(win_length),
            return_complex=True,
        )
        self.istft = lambda x: torch.istft(
            x, n_fft, hop_length=hop_length,
            window=torch.hann_window(win_length),
        )
        self.time_stretch = torchaudio.transforms.TimeStretch(
            hop_length=hop_length,
            n_freq=(n_fft//2)+1,
            fixed_rate=stretch_rate*shift_rate
        )
        self.resample = Resample(
            orig_freq=sampling_rate/shift_rate,
            new_freq=sampling_rate
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------

        Returns
        -------
        """

        spec_orig = self.stft(x)
        spec_target = self.time_stretch(spec_orig)
        x_target = self.istft(spec_target)
        return self.resample(x_target)

class Crop(torch.nn.Module):
    """
    Crop
    """

    def __init__(self,
                 waveform_length : int,
                 offset : int):
        super(Crop, self).__init__()
        self.waveform_length = waveform_length
        self.offset = offset

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        offset = self.offset % (2 * x.shape[-1])
        repeats = math.ceil((self.waveform_length + self.offset)
                            / (2 * x.shape[-1]))
        if offset + self.waveform_length <= x.shape[-1]:
            return x[..., offset:offset+self.waveform_length]
        x = torch.cat([x, x.flip(-1)] * repeats, dim=-1)
        return x[..., offset:offset+self.waveform_length]

class PadOrCrop(torch.nn.Module):
    """
    PadOrCrop

    """

    def __init__(self, waveform_length : int, random : bool = False) -> None:
        """
        Parameters
        ----------
        """

        super(PadOrCrop, self).__init__()
        self.waveform_length = waveform_length
        self.random = random

    def pad_or_crop(self, x : torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------

        Returns
        -------
        """

        orig_shape = x.shape
        offset_before=None
        if x.shape[-1] < self.waveform_length: # pad
            num_samples =\
                math.ceil(self.waveform_length / x.shape[-1])
            if self.random:
                offset = torch.randint(
                    0,
                    x.shape[-1] * num_samples - self.waveform_length - 1,
                    size=(x.shape[0],)
                )
            else:
                offset = torch.zeros((x.shape[0],), dtype=int)
            x = torch.stack([
                torch.cat([_x]*num_samples, dim=-1)
                [..., o:o+self.waveform_length]
                for _x, o in zip(x, offset)
            ], dim=0)

        elif x.shape[-1] > self.waveform_length: # crop
            if self.random:
                offset_before = torch.randint(
                    0, x.shape[-1] - self.waveform_length + 1, size=(x.shape[0],)
                )
            else:
                offset_before = torch.zeros((x.shape[0],), dtype=int)
            x = torch.stack([
                _x[..., ob:ob+self.waveform_length]
                for _x, ob in zip(x, offset_before)
            ], dim=0)

        return x

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            return self.pad_or_crop(x).squeeze(0)
        else:
            return self.pad_or_crop(x)

class TwoPointScale(torch.nn.Module):
    """
    TwoPointScale
    """

    def __init__(self,
                 scale_start : float,
                 scale_end : float,
                 transition_start : float,
                 transition_duration : float,
                 normalize : bool=False) -> None:
        """
        Parameters
        ----------
        """

        super(TwoPointScale, self).__init__()
        assert 0 <= transition_start <= 1
        assert 0 <= transition_duration <= 1

        self.scale_start = scale_start
        self.scale_end = scale_end
        self.transition_start = transition_start
        self.transition_duration = transition_duration
        self.normalize = normalize

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------

        Returns
        -------
        """

        transition_duration = int(x.shape[-1] * self.transition_duration)
        transition_start = int(
            x.shape[-1] * (1 - self.transition_duration) * self.transition_start
        )
        scale_rate = torch.cat((
            self.scale_start * torch.ones(transition_start),
            torch.linspace(self.scale_start, self.scale_end, transition_duration),
            self.scale_end * torch.ones(x.shape[-1] - transition_start - transition_duration)
        ))

        if self.normalize:
            x /= torch.max(
                torch.max(x.abs(), dim=-1, keepdims=True)[0],
                torch.ones(*x.shape[:-1], 1) * 0.1
            )
        return scale_rate * x

class MultiPointScale(torch.nn.Module):
    """
    ThreePointScale
    """

    def __init__(self,
                 scales : tp.List[float],
                 durations : tp.List[float],
                 normalize : bool=False) -> None:
        """
        Parameters
        ----------
        """

        super(MultiPointScale, self).__init__()
        assert len(scales) == len(durations) + 2

        self.scales = scales
        self.durations = [d / sum(durations) for d in durations]
        self.normalize = normalize

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------

        Returns
        -------
        """

        if len(self.durations) > 0:
            transition_length = [int(x.shape[-1] * d) for d in self.durations[:-1]]
        else:
            transition_length = []
        transition_length.append(x.shape[-1] - sum(transition_length))

        scale_rate = torch.cat([
            torch.linspace(s, t, l)
            for s, t, l in zip(
                    self.scales[:-1], self.scales[1:], transition_length)
        ])

        """
        transition_duration = int(x.shape[-1] * self.transition_duration)
        transition_start = int(
            x.shape[-1] * (1 - self.transition_duration) * self.transition_start
        )
        scale_rate = torch.cat((
            self.scale_start * torch.ones(transition_start),
            torch.linspace(self.scale_start, self.scale_end, transition_duration),
            self.scale_end * torch.ones(x.shape[-1] - transition_start - transition_duration)
        ))
        """

        if self.normalize:
            x /= torch.max(
                torch.max(x.abs(), dim=-1, keepdims=True)[0],
                torch.ones(*x.shape[:-1], 1) * 0.1
            )
        return scale_rate * x


def build_transform(orig_freq : int,
                    new_freq : int,
                    waveform_length : int,
                    time_stretch_rate : float,
                    pitch_shift_rate : float,
                    scales : tp.List[float],
                    scale_duration : tp.List[float],
                    normalize : bool=False,
                    n_fft : int=2048,
                    hop_length : int=512,
                    win_length : int=2048) -> torch.nn.Module:
    """
    Parameters
    ----------

    Returns
    -------
    """

    transforms = [
        TimeStretchAndPitchShift(orig_freq,
                                 time_stretch_rate,
                                 pitch_shift_rate,
                                 n_fft=n_fft,
                                 hop_length=hop_length,
                                 win_length=win_length),
        Resample(float(orig_freq),
                 float(new_freq)),
        MultiPointScale(scales,
                        scale_duration,
                        normalize=normalize),
    ]
    if waveform_length is not None:
        transforms.append(
            PadOrCrop(waveform_length,
                      random=False)
        )
    return Compose(transforms)
