
import typing as tp
from math import pi, log2, ceil, floor, sqrt, cos, sin
import torch

def _generate_dft_matrix(n_fft):
    phi = 2*pi*torch.arange(n_fft, dtype=torch.float) / n_fft
    basis = torch.arange(n_fft // 2 + 1, dtype=torch.float).unsqueeze(-1)
    return torch.cat((torch.cos(phi*basis), torch.sin(phi*basis)))

def _init_conv_weight(conv : torch.nn.Conv1d) -> None:
    torch.nn.init.xavier_normal_(conv.weight)
    if conv.bias is not None:
        fan_out, fan_in = \
            torch.nn.init._calculate_fan_in_and_fan_out(conv.weight)
        if (fan_in + fan_in) != 0:
            std = sqrt(2 / (fan_in + fan_out))
            torch.nn.init.normal_(conv.bias, std=std)

class TrainableStftLayer(torch.nn.Module):
    """
    Trainable stft layer
    """
    def __init__(self,
                 n_fft : int,
                 hop_length : int=None,) -> None:
        """
        n_fft : int
        hop_length : int
        """
        super(TrainableStftLayer, self).__init__()
        if hop_length is None:
            hop_length = n_fft // 4

        self.n_fft = n_fft
        self.hop_length = hop_length
        # XXX: padding amount in Conv1d
        self.conv = torch.nn.Conv1d(1,
                                    n_fft+2,
                                    n_fft,
                                    stride=hop_length,
                                    padding=hop_length * 2,
                                    bias=False,)

        weight = torch.sqrt(torch.hann_window(n_fft)) \
            * _generate_dft_matrix(n_fft)
        with torch.no_grad():
            self.conv.weight.copy_(weight.unsqueeze(1))

    def forward(self, x):
        """
        input: (batch_size, n_channels, waveform_length)
        output: (batch_size, n_channels, 2*(n_fft//2+1), time)
        """
        x_shape = x.shape[:-1]
        return self.conv(x.flatten(0, -2).unsqueeze(1))\
                   .unflatten(0, x_shape)

    def forward_length(self, l_in : int) -> int:
        return (l_in + 2 * self.hop_length*2 - self.n_fft) \
            // self.hop_length + 1

    def reverse_length(self, l_out : int) -> int:
        return self.hop_length * (l_out - 1) - 2 * self.hop_length*2 + self.n_fft

class TrainableIstftLayer(torch.nn.Module):
    def __init__(self,
                 n_fft : int,
                 hop_length : int=None,) -> None:
        super(TrainableIstftLayer, self).__init__()

        if hop_length is None:
            hop_length = n_fft // 4

        self.n_fft = n_fft
        self.hop_length = hop_length
        # XXX: padding amount in ConvTranspose1d
        self.conv = torch.nn.ConvTranspose1d(n_fft+2,
                                             1,
                                             n_fft,
                                             bias=False,
                                             stride=hop_length,
                                             padding=hop_length * 2,)

        weight = torch.sqrt(torch.hann_window(n_fft)) \
            * _generate_dft_matrix(n_fft)
        with torch.no_grad():
            self.conv.weight.copy_(weight.unsqueeze(1))

    def forward(self, x):
        """
        input: (batch_size, n_channels, 2*(n_fft//2+1), time)
        output: (batch_size, n_channels, waveform_length)
        """
        x_shape = x.shape[:-2]
        return self.conv(x.flatten(0, -3))\
                   .squeeze(-2)\
                   .unflatten(0, x_shape) / self.n_fft

    def forward_length(self, l_in : int) -> int:
        return self.hop_length * (l_in - 1) - 2 * self.hop_length*2 + self.n_fft

    def reverse_length(self, l_out : int) -> int:
        return ceil((l_out + 2 * self.hop_length*2 - self.n_fft)
                    / self.hop_length) + 1

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

        _init_conv_weight(self.in_conv)
        _init_conv_weight(self.mid_conv)
        _init_conv_weight(self.skip_conv)
        _init_conv_weight(self.out_conv)

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
        _init_conv_weight(self.mag_conv)
        _init_conv_weight(self.phase_conv)

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

        return torch.cat((
            mag * torch.cos(phase), mag * torch.sin(phase)), dim=-2)

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
        _init_conv_weight(self.conv)

    def forward(self, x : torch.Tensor):
        x = self.conv(x).unflatten(-2, (self.mask_num, -1))
        x = torch.cat((torch.sigmoid(x), torch.zeros_like(x)), dim=-2)
        return x

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

        if not mask_each:
            assert out_channel == 2
        if magbook_size > 1 or phasebook_size > 1:
            assert magbook_size > 1 and phasebook_size > 1
        assert kernel_size % 2 == 1
        n_fft = feature_channel - 2

        self.encoder = TrainableStftLayer(n_fft, hop_length)
        self.decoder = TrainableIstftLayer(n_fft, hop_length)

        self.feature_norm = torch.nn.InstanceNorm1d(feature_channel
                                                    * in_channel)
        self.in_conv = torch.nn.Conv1d(feature_channel * in_channel,
                                       bottleneck_channel,
                                       1)
        _init_conv_weight(self.in_conv)

        self.out_prelu = torch.nn.PReLU()
        if magbook_size > 1 and phasebook_size > 1:
            self.mask_module = CodebookMask(
                skipconnection_channel,
                feature_channel,
                mask_num=out_channel if mask_each else 1,
                magbook_size=magbook_size,
                phasebook_size=phasebook_size,
            )
        else:
            self.mask_module = SigmoidMask(
                skipconnection_channel,
                feature_channel,
                mask_num=out_channel if mask_each else 1
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
        F, T = base_feature.shape[-2:]

        block_in = self.in_conv(self.feature_norm(base_feature.flatten(1, -2)))

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
        base_feature = base_feature[..., None, :, :, :]
        masks = masks[..., None, :, :]
        m_re, m_im = masks[..., :F//2, :], masks[..., F//2:, :]
        b_re, b_im = base_feature[..., :F//2, :], base_feature[..., F//2:, :]
        masked_features = torch.cat((
            m_re * b_re - m_im * b_im,
            m_re * b_im + m_im * b_re,
        ), dim=-2)
        if not self.mask_each:
            masked_features = torch.cat(
                (masked_features, base_feature - masked_features),
                dim=-3
            )
        signals = self.decoder(masked_features)
        signals = signals[..., pad_left:pad_left+x.shape[-1]].squeeze(-3).squeeze(-2)

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
        _init_conv_weight(self.conv)

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

