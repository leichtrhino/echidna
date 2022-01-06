
import typing as tp
import math
import random
import resampy
import torch
import torchaudio

class Compose(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, x):
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
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------

        Returns
        -------
        """
        return torch.from_numpy(
            resampy.resample(x.cpu().numpy(), self.orig_freq, self.new_freq)
        ).to(x.device)

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
        self.window = torch.nn.parameter.Parameter(
            torch.hann_window(win_length),
            requires_grad=False
        )
        self.stft = lambda x: torch.stft(
            x, n_fft, hop_length=hop_length,
            window=self.window,
            return_complex=True,
        )
        self.istft = lambda x: torch.istft(
            x, n_fft, hop_length=hop_length,
            window=self.window,
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
        self.window = torch.nn.parameter.Parameter(
            torch.hann_window(win_length),
            requires_grad=False
        )
        self.stft = lambda x: torch.stft(
            x, n_fft, hop_length=hop_length,
            window=self.window,
            return_complex=True,
        )
        self.istft = lambda x: torch.istft(
            x, n_fft, hop_length=hop_length,
            window=self.window,
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
        self.window = torch.nn.parameter.Parameter(
            torch.hann_window(win_length),
            requires_grad=False
        )
        self.stft = lambda x: torch.stft(
            x, n_fft, hop_length=hop_length,
            window=self.window,
            return_complex=True,
        )
        self.istft = lambda x: torch.istft(
            x, n_fft, hop_length=hop_length,
            window=self.window,
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

class MultiPointScale(torch.nn.Module):
    """
    ThreePointScale
    """

    def __init__(self,
                 scales : tp.List[float],
                 fractions : tp.List[float],
                 normalize : bool=False) -> None:
        """
        Parameters
        ----------
        """

        super(MultiPointScale, self).__init__()
        assert len(scales) == len(fractions) + 1

        self.scales = scales
        self.fractions = [d / sum(fractions) for d in fractions]
        self.normalize = normalize

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------

        Returns
        -------
        """

        if len(self.fractions) > 0:
            transition_length = [int(x.shape[-1] * d) for d in self.fractions[:-1]]
        else:
            transition_length = []
        transition_length.append(x.shape[-1] - sum(transition_length))

        if len(self.scales) > 1:
            scale_rate = torch.cat([
                torch.linspace(s, t, l, device=x.device)
                for s, t, l in zip(self.scales[:-1],
                                   self.scales[1:],
                                   transition_length)
            ])
        else:
            scale_rate = self.scales[0]

        if self.normalize:
            x /= torch.max(
                torch.max(x.abs(), dim=-1, keepdims=True)[0],
                torch.ones(*x.shape[:-1], 1, device=x.device) * 0.1
            )
        return scale_rate * x


def build_transform(normalize : bool,
                    source_sample_rate : int,
                    target_sample_rate : int,
                    time_stretch_rate : int,
                    pitch_shift_rate : int,
                    scale_amounts : tp.List[float],
                    scale_fractions : tp.List[float],
                    offset : int,
                    waveform_length : int,

                    n_fft : int=2048,
                    hop_length : int=512,
                    win_length : int=2048) -> torch.nn.Module:
    """
    Parameters
    ----------

    Returns
    -------
    """

    transforms = []
    if source_sample_rate is not None \
       and target_sample_rate is not None:
        transforms.append(
            TimeStretchAndPitchShift(source_sample_rate,
                                     time_stretch_rate,
                                     pitch_shift_rate,
                                     n_fft=n_fft,
                                     hop_length=hop_length,
                                     win_length=win_length)
        )
        transforms.append(
            Resample(source_sample_rate,
                     target_sample_rate)
        )
    if waveform_length is not None and offset is not None:
        transforms.append(
            Crop(waveform_length, offset)
        )
    if scale_amounts is not None and scale_fractions is not None:
        transforms.append(
            MultiPointScale(scale_amounts,
                            scale_fractions,
                            normalize=normalize)
        )

    return Compose(transforms)

