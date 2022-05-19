
from math import pi, ceil, sqrt, cos, sin
import torch


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

