
from math import pi, ceil, sqrt, cos, sin
import torch

from .utils import init_conv_weight, generate_dft_matrix


class STFTLayer(torch.nn.Module):
    """
    Not trainable stft layer
    """
    def __init__(self,
                 n_fft : int,
                 hop_length : int=None,) -> None:
        """
        n_fft : int
        hop_length : int
        """
        super().__init__()
        if hop_length is None:
            hop_length = n_fft // 4

        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x):
        """
        input: (batch_size, n_channels, waveform_length)
        output: (batch_size, n_channels, 2, (n_fft//2), time)
        """
        window = torch.sqrt(torch.hann_window(self.n_fft, device=x.device))
        orig_shape = x.shape[:-1]

        return torch.view_as_real(
            torch.stft(x.flatten(0, -2),
                       self.n_fft,
                       self.hop_length,
                       window=window,
                       return_complex=True)
            ).unflatten(0, orig_shape)\
                    .transpose(-3, -1)\
                    .transpose(-2, -1)[..., 1:, :]

    def forward_length(self, l_in : int) -> int:
        return (l_in + 2 * self.hop_length*2 - self.n_fft) \
            // self.hop_length + 1

    def reverse_length(self, l_out : int) -> int:
        return self.hop_length * (l_out - 1) \
            - 2 * self.hop_length*2 \
            + self.n_fft

class ISTFTLayer(torch.nn.Module):
    """
    Not trainable stft layer
    """
    def __init__(self,
                 n_fft : int,
                 hop_length : int=None,) -> None:
        super().__init__()

        if hop_length is None:
            hop_length = n_fft // 4

        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x):
        """
        input: (batch_size, n_channels, 2, (n_fft//2+1), time)
        output: (batch_size, n_channels, waveform_length)
        """
        x_shape = x.shape[:-3]
        window = torch.sqrt(torch.hann_window(self.n_fft, device=x.device))
        reshaped = x.transpose(-3, -1).transpose(-3, -2).flatten(0, -4)
        re, im = torch.split(reshaped, 1, dim=-1)
        reshaped_comp = torch.squeeze(re + 1j * im, dim=-1)
        reshaped_comp = torch.cat((
            torch.zeros(
                *reshaped_comp.shape[:-2], 1, reshaped_comp.shape[-1],
                dtype=reshaped_comp.dtype, device=reshaped_comp.device
            ),
            reshaped_comp), dim=-2)
        return torch.istft(
            reshaped_comp,
            self.n_fft,
            self.hop_length,
            window=window,
        ).unflatten(0, x_shape)

    def forward_length(self, l_in : int) -> int:
        return self.hop_length * (l_in - 1) - 2 * self.hop_length*2 + self.n_fft

    def reverse_length(self, l_out : int) -> int:
        return ceil((l_out + 2 * self.hop_length*2 - self.n_fft)
                    / self.hop_length) + 1

class SigmoidMask(torch.nn.Module):
    def __init__(self,
                 in_channel : int,
                 out_channel : int,
                 mask_num : int):
        super(SigmoidMask, self).__init__()
        self.mask_num = mask_num
        self.conv = torch.nn.Conv1d(in_channel,
                                    out_channel * mask_num // 2,
                                    1)
        init_conv_weight(self.conv)

    def forward(self, x : torch.Tensor):
        x = self.conv(x).unflatten(-2, (self.mask_num, -1))
        x = torch.stack((torch.sigmoid(x), torch.zeros_like(x)), dim=-3)
        return x

class CodebookMask(torch.nn.Module):
    def __init__(self,
                 in_channel : int,
                 out_channel : int,
                 mask_num : int,
                 magbook_size : int=3,
                 phasebook_size : int=8,
                 requires_grad : bool=False,):
        super(CodebookMask, self).__init__()

        self.mag_conv = torch.nn.Conv1d(in_channel,
                                        mask_num
                                        * out_channel // 2
                                        * magbook_size,
                                        1)
        self.phase_conv = torch.nn.Conv1d(in_channel,
                                          mask_num
                                          * out_channel // 2
                                          * phasebook_size,
                                          1)
        init_conv_weight(self.mag_conv)
        init_conv_weight(self.phase_conv)

        mag_cff = torch.arange(0, magbook_size, dtype=torch.float)
        phase_cff = torch.Tensor(
            [(sin(phi), cos(phi)) for phi in # for atan2
             2*pi*torch.arange(0, phasebook_size)/phasebook_size]
        )
        self.magbook = torch.nn.Parameter(
            mag_cff.repeat(mask_num, out_channel // 2, 1),
            requires_grad=requires_grad
        )
        self.phasebook = torch.nn.Parameter(
            phase_cff.repeat(mask_num, out_channel // 2, 1, 1),
            requires_grad=requires_grad
        )

        self.mask_num = mask_num
        self.out_channel = out_channel
        self.magbook_size = magbook_size
        self.phasebook_size = phasebook_size

    def forward(self, x : torch.Tensor):
        mag = self.mag_conv(x).unflatten(
            -2, (self.mask_num, -1, self.magbook_size))
        mag = torch.nn.functional.softmax(mag, dim=-2)
        mag = torch.sum(mag * self.magbook.unsqueeze(-1), dim=-2)

        phase = self.phase_conv(x).unflatten(
            -2, (self.mask_num, -1, self.phasebook_size, 1))
        phase = torch.nn.functional.softmax(phase, dim=-3)
        phase = torch.sum(
            phase * self.phasebook.unsqueeze(-1),
            dim=-3
        )
        phase = torch.atan2(*phase.split(1, dim=-2)).squeeze(-2)

        return torch.stack((
            mag * torch.cos(phase), mag * torch.sin(phase)), dim=-3)

