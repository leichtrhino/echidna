
import typing as tp
import torch

from .utils import init_conv_weight
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

class ConvTasNet(torch.nn.Module):
    def __init__(self,
                 in_channel : int,
                 out_channel : int,
                 feature_channel : int,
                 block_channel : int,
                 bottleneck_channel : int,
                 skipconnection_channel : int,
                 kernel_size : int,
                 depth : int,
                 repeats : int,
                 magbook_size : int=1,
                 phasebook_size : int=1,
                 mask_each : bool=True,
                 hop_length=None) -> None:
        super(ConvTasNet, self).__init__()

        if magbook_size > 1 or phasebook_size > 1:
            assert magbook_size > 1 and phasebook_size > 1
        assert kernel_size % 2 == 1
        n_fft = feature_channel

        self.encoder = TrainableSTFTLayer(n_fft, hop_length)
        self.decoder = TrainableISTFTLayer(n_fft, hop_length)

        self.feature_norm = torch.nn.InstanceNorm1d(feature_channel
                                                    * in_channel)
        self.in_conv = torch.nn.Conv1d(feature_channel * in_channel,
                                       bottleneck_channel,
                                       1)
        init_conv_weight(self.in_conv)

        self.out_prelu = torch.nn.PReLU()
        mask_num = in_channel \
            * (out_channel if mask_each else out_channel - 1)
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

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.skipconnection_channel = skipconnection_channel
        self.depth = depth
        self.mask_each = mask_each

    def forward(self,
                x : torch.Tensor,
                return_embd : bool=False) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        l_in = self.encoder.reverse_length(
            self.decoder.reverse_length(x.shape[-1]))
        pad_left = (l_in - x.shape[-1]) // 2
        pad_right = l_in - x.shape[-1] - pad_left
        x_ = torch.cat((torch.zeros(*x.shape[:-1], pad_left, device=x.device),
                        x,
                        torch.zeros(*x.shape[:-1], pad_right, device=x.device)),
                       dim=-1)

        # encode
        base_feature = self.encoder(x_)

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

        embd = self.out_prelu(global_residual)
        masks = self.mask_module(embd)

        # decode from masks
        base_feature = base_feature.unsqueeze(-5)
        masks = masks.unflatten(
            -4,
            (self.out_channel if self.mask_each else self.out_channel - 1,
             self.in_channel)
        )
        m_re, m_im = [m.squeeze(-3) for m in masks.split(1, dim=-3)]
        b_re, b_im = [m.squeeze(-3) for m in base_feature.split(1, dim=-3)]
        masked_features = torch.stack((
            m_re * b_re - m_im * b_im,
            m_re * b_im + m_im * b_re,
        ), dim=-3)

        if not self.mask_each:
            other_feature = base_feature.squeeze(-5) \
                - torch.sum(masked_features, dim=-5)
            masked_features = torch.cat((
                masked_features,
                other_feature.unsqueeze(-5)
            ), dim=-5)
        signals = self.decoder(masked_features)
        signals = signals[..., pad_left:pad_left+x.shape[-1]]\
            .squeeze(-3).squeeze(-2)

        if return_embd:
            return signals, embd
        else:
            return signals


    def forward_length(self, l_in : int) -> int:
        return l_in

    def reverse_length(self, l_out : int) -> int:
        return l_out

    def forward_embd_feature(self) -> int:
        return self.skipconnection_channel

    def forward_embd_length(self, l_in : int) -> int:
        l_in = self.encoder.reverse_length(self.decoder.reverse_length(l_in))
        return self.encoder.forward_length(l_in)

    def parameter_list(self, base_lr):
        return [
            {'params': self.encoder.parameters(), 'lr': base_lr * 1e-3},
            {'params': self.feature_norm.parameters()},
            {'params': self.in_conv.parameters()},
            {'params': self.blocks.parameters()},
            {'params': self.block_scales.parameters()},
            {'params': self.out_prelu.parameters()},
            {'params': self.mask_module.parameters()},
            {'params': self.decoder.parameters(), 'lr': base_lr * 1e-3},
        ]

class ChimeraConvTasNet(torch.nn.Module):
    """
    ChimeraConvTasNet module
    """
    def __init__(self,
                 in_channel : int,
                 out_channel : int,
                 feature_channel : int,
                 block_channel : int,
                 bottleneck_channel : int,
                 skipconnection_channel : int,
                 embd_feature : int,
                 embd_dim : int,
                 kernel_size : int,
                 depth : int,
                 repeats : int,
                 magbook_size : int=1,
                 phasebook_size : int=1,
                 mask_each : bool=True,
                 hop_length : int=None) -> None:
        """
        Parameter
        ---------

        """
        super(ChimeraConvTasNet, self).__init__()

        self.convtasnet = ConvTasNet(
            in_channel=in_channel,
            out_channel=out_channel,
            feature_channel=feature_channel,
            block_channel=block_channel,
            bottleneck_channel=bottleneck_channel,
            skipconnection_channel=skipconnection_channel,
            kernel_size=kernel_size,
            depth=depth,
            repeats=repeats,
            magbook_size=magbook_size,
            phasebook_size=phasebook_size,
            mask_each=mask_each,
            hop_length=hop_length
        )
        self.conv = torch.nn.Conv1d(self.convtasnet.forward_embd_feature(),
                                    embd_feature * embd_dim,
                                    1)
        init_conv_weight(self.conv)

        self.embd_feature = embd_feature
        self.embd_dim = embd_dim

    def forward(self,
                x : torch.Tensor) -> torch.Tensor:
        """
        Parameter
        ---------
        x : torch.Tensor
        """
        y, embd = self.convtasnet(x, return_embd=True)
        embd = self.conv(embd)\
                   .unflatten(-2, (self.embd_feature, self.embd_dim))\
                   .transpose(-1, -2)\
                   .sigmoid()
        return y, embd / embd.norm(dim=-1, keepdim=True)

    def forward_length(self, l_in : int) -> int:
        """
        Parameter
        ---------
        l_in : int
        """
        return self.convtasnet.forward_length(l_in)

    def reverse_length(self, l_out : int) -> int:
        """
        Parameter
        ---------
        l_out : int
        """
        return self.convtasnet.reverse_length(l_out)

    def forward_embd_feature(self) -> int:
        return self.embd_feature

    def forward_embd_length(self, l_in : int) -> int:
        return self.convtasnet.forward_embd_length(l_in)

    def get_core_model(self) -> torch.nn.Module:
        return self.convtasnet

    def parameter_list(self, base_lr):
        return self.convtasnet.parameter_list(base_lr)\
            + [{'params': self.conv.parameters()}]

class ConvTasNetEncoder(torch.nn.Module):
    def __init__(self,
                 in_channel : int,
                 out_channel : int,
                 feature_channel : int,
                 block_channel : int,
                 bottleneck_channel : int,
                 skipconnection_channel : int,
                 kernel_size : int,
                 depth : int,
                 repeats : int,
                 magbook_size : int=1,
                 phasebook_size : int=1,
                 mask_each : bool=True,
                 hop_length=None) -> None:
        super(ConvTasNet, self).__init__()

        if magbook_size > 1 or phasebook_size > 1:
            assert magbook_size > 1 and phasebook_size > 1
        assert kernel_size % 2 == 1
        n_fft = feature_channel

        self.stft = TrainableSTFTLayer(n_fft, hop_length)
        self.istft = TrainableISTFTLayer(n_fft, hop_length)

        self.feature_norm = torch.nn.InstanceNorm1d(feature_channel
                                                    * in_channel)
        self.in_conv = torch.nn.Conv1d(feature_channel * in_channel,
                                       bottleneck_channel,
                                       1)
        init_conv_weight(self.in_conv)

        self.out_prelu = torch.nn.PReLU()
        mask_num = in_channel \
            * (out_channel if mask_each else out_channel - 1)
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

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.skipconnection_channel = skipconnection_channel
        self.depth = depth
        self.mask_each = mask_each

    def forward(self,
                x : torch.Tensor,
                return_embd : bool=False) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        l_in = self.stft.reverse_length(
            self.istft.reverse_length(x.shape[-1]))
        pad_left = (l_in - x.shape[-1]) // 2
        pad_right = l_in - x.shape[-1] - pad_left
        x_ = torch.cat((torch.zeros(*x.shape[:-1], pad_left, device=x.device),
                        x,
                        torch.zeros(*x.shape[:-1], pad_right, device=x.device)),
                       dim=-1)

        # encode
        base_feature = self.stft(x_)

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

        embd = self.out_prelu(global_residual)
        masks = self.mask_module(embd)

        # decode from masks
        base_feature = base_feature.unsqueeze(-5)
        masks = masks.unflatten(
            -4,
            (self.out_channel if self.mask_each else self.out_channel - 1,
             self.in_channel)
        )
        m_re, m_im = [m.squeeze(-3) for m in masks.split(1, dim=-3)]
        b_re, b_im = [m.squeeze(-3) for m in base_feature.split(1, dim=-3)]
        masked_features = torch.stack((
            m_re * b_re - m_im * b_im,
            m_re * b_im + m_im * b_re,
        ), dim=-3)

        if not self.mask_each:
            other_feature = base_feature.squeeze(-5) \
                - torch.sum(masked_features, dim=-5)
            masked_features = torch.cat((
                masked_features,
                other_feature.unsqueeze(-5)
            ), dim=-5)
        signals = self.istft(masked_features)
        signals = signals[..., pad_left:pad_left+x.shape[-1]]\
            .squeeze(-3).squeeze(-2)

        if return_embd:
            return signals, embd
        else:
            return signals


    def forward_length(self, l_in : int) -> int:
        return l_in

    def reverse_length(self, l_out : int) -> int:
        return l_out

    def forward_embd_feature(self) -> int:
        return self.skipconnection_channel

    def forward_embd_length(self, l_in : int) -> int:
        l_in = self.stft.reverse_length(self.istft.reverse_length(l_in))
        return self.stft.forward_length(l_in)

    def parameter_list(self, base_lr):
        return [
            {'params': self.stft.parameters(), 'lr': base_lr * 1e-3},
            {'params': self.feature_norm.parameters()},
            {'params': self.in_conv.parameters()},
            {'params': self.blocks.parameters()},
            {'params': self.block_scales.parameters()},
            {'params': self.out_prelu.parameters()},
            {'params': self.mask_module.parameters()},
            {'params': self.istft.parameters(), 'lr': base_lr * 1e-3},
        ]


class ConvTasNetDecoder(torch.nn.Module):
    def __init__(self,
                 in_channel : int,
                 out_channel : int,
                 feature_channel : int,
                 block_channel : int,
                 bottleneck_channel : int,
                 skipconnection_channel : int,
                 kernel_size : int,
                 depth : int,
                 repeats : int,
                 magbook_size : int=1,
                 phasebook_size : int=1,
                 mask_each : bool=True,
                 hop_length=None) -> None:
        super(ConvTasNet, self).__init__()

        if magbook_size > 1 or phasebook_size > 1:
            assert magbook_size > 1 and phasebook_size > 1
        assert kernel_size % 2 == 1
        n_fft = feature_channel

        self.stft = TrainableSTFTLayer(n_fft, hop_length)

        # decoder
        self.istft = TrainableISTFTLayer(n_fft, hop_length)

        self.feature_norm = torch.nn.InstanceNorm1d(feature_channel
                                                    * in_channel)
        self.in_conv = torch.nn.Conv1d(feature_channel * in_channel,
                                       bottleneck_channel,
                                       1)
        init_conv_weight(self.in_conv)

        self.out_prelu = torch.nn.PReLU()

        # decoder
        mask_num = in_channel \
            * (out_channel if mask_each else out_channel - 1)
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

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.skipconnection_channel = skipconnection_channel
        self.depth = depth
        self.mask_each = mask_each

    def forward(self,
                x : torch.Tensor,
                return_embd : bool=False) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        l_in = self.stft.reverse_length(
            self.istft.reverse_length(x.shape[-1]))
        pad_left = (l_in - x.shape[-1]) // 2
        pad_right = l_in - x.shape[-1] - pad_left
        x_ = torch.cat((torch.zeros(*x.shape[:-1], pad_left, device=x.device),
                        x,
                        torch.zeros(*x.shape[:-1], pad_right, device=x.device)),
                       dim=-1)

        # encode
        base_feature = self.stft(x_)

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

        embd = self.out_prelu(global_residual)
        masks = self.mask_module(embd)

        # decode from masks
        base_feature = base_feature.unsqueeze(-5)
        masks = masks.unflatten(
            -4,
            (self.out_channel if self.mask_each else self.out_channel - 1,
             self.in_channel)
        )
        m_re, m_im = [m.squeeze(-3) for m in masks.split(1, dim=-3)]
        b_re, b_im = [m.squeeze(-3) for m in base_feature.split(1, dim=-3)]
        masked_features = torch.stack((
            m_re * b_re - m_im * b_im,
            m_re * b_im + m_im * b_re,
        ), dim=-3)

        if not self.mask_each:
            other_feature = base_feature.squeeze(-5) \
                - torch.sum(masked_features, dim=-5)
            masked_features = torch.cat((
                masked_features,
                other_feature.unsqueeze(-5)
            ), dim=-5)
        signals = self.istft(masked_features)
        signals = signals[..., pad_left:pad_left+x.shape[-1]]\
            .squeeze(-3).squeeze(-2)

        if return_embd:
            return signals, embd
        else:
            return signals


    def forward_length(self, l_in : int) -> int:
        return l_in

    def reverse_length(self, l_out : int) -> int:
        return l_out

    def forward_embd_feature(self) -> int:
        return self.skipconnection_channel

    def forward_embd_length(self, l_in : int) -> int:
        l_in = self.stft.reverse_length(self.istft.reverse_length(l_in))
        return self.stft.forward_length(l_in)

    def parameter_list(self, base_lr):
        return [
            {'params': self.stft.parameters(), 'lr': base_lr * 1e-3},
            {'params': self.feature_norm.parameters()},
            {'params': self.in_conv.parameters()},
            {'params': self.blocks.parameters()},
            {'params': self.block_scales.parameters()},
            {'params': self.out_prelu.parameters()},
            {'params': self.mask_module.parameters()},
            {'params': self.istft.parameters(), 'lr': base_lr * 1e-3},
        ]


