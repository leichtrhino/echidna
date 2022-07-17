
import typing as tp
import torch

from ..utils import init_conv_weight
from .commonlayers import (
    STFTLayer,
    ISTFTLayer,
    TrainableSTFTLayer,
    TrainableISTFTLayer,
    SigmoidMask,
    CodebookMask,
)

class ConvBlock(torch.nn.Module):
    def __init__(self,
                 io_channel : int,
                 block_channel : int,
                 skipconnection_channel : int,
                 kernel_size : int,
                 dilation : int,) -> None:
        super(ConvBlock, self).__init__()

        in_channel, out_channel = io_channel, io_channel

        self.in_conv = torch.nn.Conv1d(in_channel,
                                       block_channel,
                                       1,
                                       bias=False)
        self.in_relu = torch.nn.PReLU()
        self.in_norm = torch.nn.InstanceNorm1d(block_channel)
        self.mid_conv = torch.nn.Conv1d(block_channel,
                                        block_channel,
                                        kernel_size,
                                        padding=(kernel_size-1)//2*dilation,
                                        dilation=dilation,
                                        groups=block_channel,
                                        bias=False)
        self.out_relu = torch.nn.PReLU()
        self.out_norm = torch.nn.InstanceNorm1d(block_channel)
        self.skip_conv = torch.nn.Conv1d(block_channel,
                                         skipconnection_channel,
                                         1)
        self.out_conv = torch.nn.Conv1d(block_channel,
                                        out_channel,
                                        1)

        init_conv_weight(self.in_conv)
        init_conv_weight(self.mid_conv)
        init_conv_weight(self.skip_conv)
        init_conv_weight(self.out_conv)

    def forward(self, x):
        b_in = self.in_norm(self.in_relu(self.in_conv(x)))
        b_out = self.out_norm(self.out_relu(self.mid_conv(b_in)))
        sc = self.skip_conv(b_out)
        out = self.out_conv(b_out)
        return sc, out + x

    def forward_length(self, l_in : int) -> int:
        return l_in

    def reverse_length(self, l_out : int) -> int:
        return l_out


class ConvTasNetEncoder(torch.nn.Module):
    def __init__(self,
                 encoder_in_channel : int,
                 feature_channel : int,
                 block_channel : int,
                 bottleneck_channel : int,
                 skipconnection_channel : int,
                 kernel_size : int,
                 depth : int,
                 repeats : int,
                 hop_length=None) -> None:
        super().__init__()

        assert kernel_size % 2 == 1

        self.encoder_in_channel = encoder_in_channel
        self.skipconnection_channel = skipconnection_channel
        self.depth = depth

        self.stft = TrainableSTFTLayer(feature_channel, hop_length)
        self.feature_norm = torch.nn.InstanceNorm1d(
            feature_channel * encoder_in_channel)
        self.in_conv = torch.nn.Conv1d(
            feature_channel * encoder_in_channel,
            bottleneck_channel,
            1)
        init_conv_weight(self.in_conv)

        self.blocks = torch.nn.ModuleList()
        for ri in range(repeats):
            for bi in range(depth):
                self.blocks.append(ConvBlock(bottleneck_channel,
                                             block_channel,
                                             skipconnection_channel,
                                             kernel_size,
                                             dilation=2**bi))
        self.block_scales = torch.nn.ParameterList([
            torch.nn.Parameter(torch.Tensor([0.9**l]))
            for l in range(depth * repeats)
        ])

    def forward(self, x : torch.Tensor) \
        -> tp.Tuple[torch.Tensor, torch.Tensor]:

        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # encode
        base_feature = self.stft(x)

        block_in = self.in_conv(self.feature_norm(
            base_feature.flatten(1, -2)))

        # calculate residual
        global_residual = None
        for li, (s, l) in enumerate(zip(self.block_scales, self.blocks), 1):
            sc, o = l(block_in)

            if global_residual is None:
                global_residual = s * sc
            else:
                global_residual = global_residual + s * sc

            if li % self.depth == 0:
                block_in = o
            else:
                block_in = block_in + o
        embd = global_residual

        return embd, base_feature

    def forward_length(self, l_in : int) -> int:
        return self.stft.forward_length(l_in)

    def reverse_length(self, l_out : int) -> int:
        return self.stft.reverse_length(l_out)

    def forward_channel(self) -> int:
        return self.skipconnection_channel

    def parameter_list(self, base_lr):
        return [
            {'params': self.stft.parameters(), 'lr': base_lr * 1e-3},
            {'params': self.feature_norm.parameters()},
            {'params': self.in_conv.parameters()},
            {'params': self.blocks.parameters()},
            {'params': self.block_scales.parameters()},
        ]


class ConvTasNetDecoder(torch.nn.Module):
    def __init__(self,
                 encoder_in_channel : int,
                 decoder_out_channel : int,
                 feature_channel : int,
                 skipconnection_channel : int,
                 magbook_size : int=1,
                 phasebook_size : int=1,
                 output_residual : bool=False,
                 hop_length=None
                 ) -> None:

        super().__init__()

        if magbook_size > 1 or phasebook_size > 1:
            assert magbook_size > 1 and phasebook_size > 1

        self.encoder_in_channel = encoder_in_channel
        self.decoder_out_channel = decoder_out_channel
        self.skipconnection_channel = skipconnection_channel
        self.output_residual = output_residual

        self.out_prelu = torch.nn.PReLU()

        # initialize mask module
        mask_num = encoder_in_channel \
            * (decoder_out_channel - 1 if output_residual
               else decoder_out_channel)
        if magbook_size > 1 and phasebook_size > 1:
            self.mask_module = CodebookMask(
                skipconnection_channel,
                feature_channel,
                mask_num=mask_num,
                magbook_size=magbook_size,
                phasebook_size=phasebook_size,
            )
        else:
            self.mask_module = SigmoidMask(
                skipconnection_channel,
                feature_channel,
                mask_num=mask_num
            )

        self.istft = TrainableISTFTLayer(feature_channel, hop_length)

    def forward(self,
                embd : torch.Tensor,
                base_feature : torch.Tensor) -> torch.Tensor:

        masks = self.mask_module(self.out_prelu(embd))

        # decode from masks
        masks = masks.unflatten(
            -4,
            (self.decoder_out_channel - 1 if self.output_residual
             else self.decoder_out_channel,
             self.encoder_in_channel)
        )
        m_re, m_im = [m.squeeze(-3) for m in masks.split(1, dim=-3)]
        b_re, b_im = [m.squeeze(-3) for m in base_feature.unsqueeze(-5).split(1, dim=-3)]
        masked_features = torch.stack((
            m_re * b_re - m_im * b_im,
            m_re * b_im + m_im * b_re,
        ), dim=-3)

        if self.output_residual:
            other_feature = base_feature \
                - torch.sum(masked_features, dim=-5)
            masked_features = torch.cat((
                masked_features,
                other_feature.unsqueeze(-5)
            ), dim=-5)

        signals = self.istft(masked_features)
        signals = signals.squeeze(-3).squeeze(-2)

        return signals

    def forward_length(self, l_in : int) -> int:
        return self.istft.forward_length(l_in)

    def reverse_length(self, l_out : int) -> int:
        return self.istft.reverse_length(l_out)

    def parameter_list(self, base_lr):
        return [
            {'params': self.out_prelu.parameters()},
            {'params': self.mask_module.parameters()},
            {'params': self.istft.parameters(), 'lr': base_lr * 1e-3},
        ]


