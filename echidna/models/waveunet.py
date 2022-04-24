
import typing as tp
from math import log2, ceil
import torch

from .utils import init_conv_weight, match_length

class DownsamplingBlock(torch.nn.Module):
    """
    Performs convolution and downsampling
    """
    def __init__(self,
                 channel_in : int,
                 channel_out : int,
                 kernel_size : int,
                 downsample_rate : int) -> None:
        """

        Parameters
        ----------
            channel_in : int
            channel_out : int
            kernel_size : int
            downsample_rate : int
        """
        super(DownsamplingBlock, self).__init__()
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.downsample_rate = downsample_rate
        self.conv = torch.nn.Conv1d(
            channel_in, channel_out, kernel_size)
        init_conv_weight(self.conv)

    def forward(self, x : torch.Tensor) -> tp.Tuple[torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            the shape should be (N, C_in, L)

        Returns
        -------
        torch.Tensor
            downsampled tensor of shape (N, C_out, ceil(L/downsample_rate))
        """
        c_out = self.conv(x)
        out = c_out[..., ::self.downsample_rate]
        return out, c_out

    def forward_length(self, l_in : int) -> int:
        """
        Parameter
        ---------
        l_in : int

        Returns
        -------
        int
            output length
        """
        return (l_in - (self.kernel_size - 1) - 1) // self.downsample_rate + 1

    def reverse_length(self, l_out : int) -> int:
        """
        Parameter
        ---------
        l_out : int
            desired output length

        Returns
        -------
        int
            minimum input length that satisfies block output of l_out
        """
        return self.downsample_rate * (l_out - 1) \
            + self.kernel_size - 1 + 1

class UpsamplingBlock(torch.nn.Module):
    """
    Performs upsample
    """
    def __init__(self,
                 channel_in : int,
                 channel_residual : int,
                 channel_out : int,
                 kernel_size : int,
                 upsample_rate : int,
                 interpolation_mode : str='nearest'):
        """

        Parameters
        ----------
        channel_in : int
        channel_out : int
        kernel_size : int
        upsample_rate : int
        interpolation_mode : str
        """
        super(UpsamplingBlock, self).__init__()
        assert kernel_size % 2 == 1
        assert upsample_rate == 2**(int(log2(upsample_rate)))
        self.kernel_size = kernel_size
        self.upsample_rate = upsample_rate

        self.upsample = Interpolation(
            upsample_rate, interpolation_mode, channel_in)
        self.conv = torch.nn.Conv1d(
            channel_in + channel_residual, channel_out, kernel_size)
        init_conv_weight(self.conv)

    def forward(self,
                x : torch.Tensor,
                residual : torch.Tensor) -> torch.Tensor:
        """
        Parameter
        ---------
        x : torch.Tensor
            tensor of shape (N, C_in, L)
        residual : torch.Tensor

        Returns
        -------
        torch.Tensor
            tensor of shape (N, C_out, (L-1)*r+1-k+1)
        """
        us_out = self.upsample(x)
        us_out = torch.cat(
            (us_out, match_length(residual, us_out.shape[-1])), dim=1)
        conv_out = self.conv(us_out)
        return conv_out

    def forward_length(self, l_in : int) -> int:
        """
        Parameter
        ---------
        l_in : int

        Returns
        -------
        int
            output length
        """
        l_out = self.upsample.forward_length(l_in)
        return l_out - self.kernel_size + 1

    def reverse_length(self, l_out : int) -> int:
        """
        Parameter
        ---------
        l_out : int
            desired output length

        Returns
        -------
        int
            minimum input length that satisfies output of l_out
        """
        l_conv_out = l_out + self.kernel_size - 1
        return self.upsample.reverse_length(l_conv_out)

class Interpolation(torch.nn.Module):
    """
    Performs interpolation.

    """
    def __init__(self,
                 upsample_rate : int,
                 mode : str='nearest',
                 channel : int=None):
        """

        Parameters
        ----------
        upsample_rate : int
        mode : str
            'nearest', 'linear', 'trinable'
        channel : int
            number of features (activated if mode == 'trainable')
        """
        super(Interpolation, self).__init__()

        if mode not in ('nearest', 'linear', 'trainable'):
            raise NotImplementedError(f'mode {mode} is not implemented')
        if upsample_rate != 2**(int(log2(upsample_rate))):
            raise ValueError(f'upsample_rate must be power of 2')
        if mode == 'trainable' and channel is None:
            raise ValueError(f'channel must be given if mode==trainable')

        self.channel = channel
        self.upsample_rate = upsample_rate
        self.mode = mode

        if self.mode == 'trainable':
            self.weight = torch.nn.parameter.Parameter(
                torch.zeros(upsample_rate-1, channel), requires_grad=True)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x : torch.Tensor
            the shape must be (N, C, L)

        Returns
        -------
        torch.Tensor
            interpolated tensor of shape (N, C, (L-1)*upsample_rate+1)
        """
        n = self.upsample_rate

        ys = [None] * (n+1)
        ys[0] = x[..., :-1]
        ys[n] = x[..., 1:]

        # upsample = 2: 1(0,2)
        # upsample = 4: 2(0,4), 1(0,2), 3(2,4)
        # upsample = 8: 4(0,8), 2(0,4), 6(4,8), 1(0,2), 3(2,4), 5, 7
        # init: upsample_rate // 2, step=rate//2
        # for each branch with n:
        #   interpolate n-step, n+step
        #   push n-step//2, n+step//2 and step=step//2 if step//2 > 0
        stack = [(n // 2, n // 2)]
        while stack:
            idx, step = stack.pop()
            if self.mode == 'trainable':
                s = torch.sigmoid(self.weight[idx-1])[None, ..., None]
            elif self.mode == 'linear':
                s = 0.5
            elif self.mode == 'nearest':
                s = 1.0 if idx < n // 2 else 0.0 if idx > n // 2 else 0.5
            ys[idx] = s * ys[idx-step] + (1-s) * ys[idx+step]
            if step // 2 > 0:
                stack.append((idx - step // 2, step // 2))
                stack.append((idx + step // 2, step // 2))

        y = torch.cat(ys, dim=-1)[..., :n*(x.shape[-1]-1)+1]
        return y

    def forward_length(self, l_in : int) -> int:
        """
        Parameter
        ---------
        l_in : int

        Returns
        -------
        int
            output length
        """
        return (l_in - 1) * self.upsample_rate + 1


    def reverse_length(self, l_out : int) -> int:
        """
        Parameter
        ---------
        l_out : int
            desired output length

        Returns
        -------
        int
            minimum input length that satisfies layer(l_in) >= l_out
        """
        return ceil((l_out - 1) / self.upsample_rate) + 1


class WaveUNetEncoder(torch.nn.Module):
    """
    Encoder module for wave u net
    """
    def __init__(self,
                 channel_in : int,
                 downsampling_channel_out : tp.List[int],
                 downsampling_kernel_size : int,
                 downsampling_rate : int,
                 encoder_channel_out : tp.List[int],
                 encoder_kernel_size : int,
                 ) -> None:
        """
        Parameter
        ---------
        channel_in : int
        downsampling_channel_out : List[int]
        downsampling_kernel_size : int
        downsampling_rate : int
        encoder_channel_out : List[int]
        encoder_kernel_size : int
        """

        super().__init__()
        self.channel_in = channel_in
        self.downsampling_channel_out = downsampling_channel_out
        self.downsampling_kernel_size = downsampling_kernel_size
        self.downsampling_rate = downsampling_rate
        self.encoder_channel_out = encoder_channel_out
        self.encoder_kernel_size = encoder_kernel_size

        self.downsampling_layers = torch.nn.ModuleList()
        c_in = channel_in
        for c_out in downsampling_channel_out:
            self.downsampling_layers.append(
                DownsamplingBlock(c_in,
                                  c_out,
                                  downsampling_kernel_size,
                                  downsampling_rate)
            )
            c_in = c_out

        self.encoder_layers = torch.nn.ModuleList()
        for c_out in encoder_channel_out:
            encoder_layer = torch.nn.Conv1d(c_in,
                                            c_out,
                                            encoder_kernel_size)
            init_conv_weight(encoder_layer)
            self.encoder_layers.append(encoder_layer)
            c_in = c_out


    def forward(self, x : torch.Tensor) \
        -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        """
        Perform encoding

        Parameter
        ---------
        x : torch.Tensor
        returns_each : bool

        Returns
        -------
        torch.Tensor
        List[torch.Tensor]
        """

        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        residual_out = []
        ds_out = x
        for l in self.downsampling_layers:
            ds_out, residual = l(ds_out)
            residual_out.append(residual)

        enc_out = ds_out
        for l in self.encoder_layers:
            enc_out = l(enc_out)

        return enc_out, residual_out, x

    def forward_length(self, l_in : int) -> int:
        """
        Parameter
        ---------
        l_in : int

        Returns
        -------
        int
        """
        #for b in self.layers[:-1]:
        for b in self.downsampling_layers:
            l_in = b.forward_length(l_in)
        return l_in \
            - (self.encoder_kernel_size - 1) \
            * len(self.encoder_layers)


    def reverse_length(self, l_out : int) -> int:
        """
        Parameter
        ---------
        l_out : int

        Returns
        -------
        int
        """
        l_out = l_out \
            + (self.encoder_kernel_size - 1) \
            * len(self.encoder_layers)

        for b in self.downsampling_layers[::-1]:
            l_out = b.reverse_length(l_out)
        return l_out

    def forward_feature_size(self) -> int:
        """
        Parameter
        ---------

        Returns
        -------
        int
        """
        return self.encoder_channel_out[-1]

    def parameter_list(self, base_lr : float):
        return [
            {'params': self.downsampling_layers.parameters()},
            {'params': self.encoder_layers.parameters()},
        ]


class WaveUNetDecoder(torch.nn.Module):
    """
    Encoder module for wave u net
    """
    def __init__(self,
                 upsampling_channel_in : int,
                 upsampling_channel_out : tp.List[int],
                 upsampling_kernel_size : int,
                 upsampling_rate : int,
                 upsampling_mode : str,
                 upsampling_residual_channel : tp.List[int],
                 decoder_channel_out : int,
                 decoder_kernel_size : int,
                 decoder_residual_channel : int) -> None:
        """
        Parameter
        ---------
        upsampling_channel_in : int
        upsampling_channel_out : List[int]
        upsampling_kernel_size : int
        upsampling_rate : int
        upsampling_mode : str
        upsampling_residual_channel : List[int]
        decoder_channel_out : int,
        decoder_kernel_size : int,
        decoder_residual_channel : int
        """

        super().__init__()
        assert upsampling_residual_channel \
            and len(upsampling_residual_channel) == len(upsampling_channel_out)

        self.upsampling_kernel_size = upsampling_kernel_size
        self.decoder_kernel_size = decoder_kernel_size

        self.upsampling_layers = torch.nn.ModuleList()
        c_in = upsampling_channel_in
        for c_out, c_residual in \
            zip(upsampling_channel_out, upsampling_residual_channel):
            self.upsampling_layers.append(
                UpsamplingBlock(c_in,
                                c_residual,
                                c_out,
                                upsampling_kernel_size,
                                upsampling_rate,
                                upsampling_mode)
            )
            c_in = c_out

        self.decoder_layer = torch.nn.Conv1d(
            c_in + decoder_residual_channel,
            decoder_channel_out,
            decoder_kernel_size
        )


    def forward(self,
                x : torch.Tensor,
                upsampling_residual : tp.List[torch.Tensor],
                decoder_residual) -> torch.Tensor:
        """
        Perform decoding

        Parameter
        ---------
        xesidual : torch.Tensor
        upsampling_residual : tp.List[torch.Tensor]
        decoder_residual : tp.List[torch.Tensor

        Returns
        -------
        torch.Tensor : returns_each=False
        List[torch.Tensor] : returns_each=True
        """

        out = x
        for l, s in zip(self.upsampling_layers, upsampling_residual[::-1]):
            out = l(out, s)
        out = self.decoder_layer(torch.cat((
            out, match_length(decoder_residual, out.shape[-1])
        ), dim=1))
        return out

    def forward_length(self, l_in : int) -> int:
        """
        Parameter
        ---------
        l_in : int

        Returns
        -------
        int
        """

        for l in self.upsampling_layers:
            l_in = l.forward_length(l_in)
        l_in = l_in - self.decoder_kernel_size + 1
        return l_in

    def reverse_length(self, l_out : int) -> int:
        """
        Parameter
        ---------
        l_out : int

        Returns
        -------
        int
        """

        l_out = l_out + self.decoder_kernel_size - 1
        for l in self.upsampling_layers[::-1]:
            l_out = l.reverse_length(l_out)
        return l_out

    def parameter_list(self, base_lr : float) -> tp.List:
        return [
            {'params': self.upsampling_layers.parameters()},
            {'params': self.decoder_layer.parameters()}
        ]


