
import typing as tp
from math import log2, ceil, floor, sqrt
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





class Encoder(torch.nn.Module):
    """
    Encoder module for wave u net
    """
    def __init__(self,
                 channel_in : int,
                 channel_out : tp.List[int],
                 kernel_size : int,
                 downsample_rate : int=2,) -> None:
        """
        Parameter
        ---------
        channel_in : int
        channel_out : List[int]
        kernel_size : int
        downsample_rate : int
        """
        super(Encoder, self).__init__()
        self.kernel_size = kernel_size
        self.layers = torch.nn.ModuleList([])
        c_in = channel_in
        #for c_out in channel_out[:-1]:
        for c_out in channel_out:
            self.layers.append(
                DownsamplingBlock(c_in,
                                  c_out,
                                  kernel_size,
                                  downsample_rate)
            )
            c_in = c_out
        """
        self.layers.append(
            torch.nn.Conv1d(c_in, channel_out[-1], kernel_size))
        """

    def forward(self, x : torch.Tensor, return_each : bool=False) \
        -> tp.Union[torch.Tensor, tp.List[torch.Tensor]]:
        """
        Perform encoding

        Parameter
        ---------
        x : torch.Tensor
        returns_each : bool

        Returns
        -------
        torch.Tensor : returns_each=False
        List[torch.Tensor] : returns_each=True
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        out_list = []
        for l in self.layers:
            out_list.append(l(x if not out_list else out_list[-1]))

        return out_list if return_each else out_list[-1]

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
        for b in self.layers:
            l_in = b.forward_length(l_in)
        return l_in
        #return l_in - self.kernel_size + 1 # for Conv1d

    def reverse_length(self, l_out : int) -> int:
        """
        Parameter
        ---------
        l_out : int

        Returns
        -------
        int
        """
        #l_out = l_out + self.kernel_size - 1 # for Conv1d
        #for b in self.layers[-2::-1]:
        for b in self.layers[::-1]:
            l_out = b.reverse_length(l_out)
        return l_out


class Decoder(torch.nn.Module):
    """
    Encoder module for wave u net
    """
    def __init__(self,
                 channel_in : int,
                 channel_out : tp.List[int],
                 kernel_size : int,
                 upsample_rate : int=2,
                 upsample_mode : str='nearest',
                 residual_mode : str='none',
                 residual_channel : tp.List[int]=None) -> None:
        """
        Parameter
        ---------
        channel_in : int
        channel_out : List[int]
        kernel_size : int
        upsample_rate : int
        residual_mode : str
            must be 'none', 'concat', or 'add'
        residual_channel : int
            extra input channels for each input (activated if mode=='concat)
        """
        super(Decoder, self).__init__()
        assert residual_mode in ('none', 'concat', 'add')
        if residual_mode == 'concat':
            assert residual_channel \
                and len(residual_channel) == len(channel_out) - 1
        else:
            residual_channel = [0 for _ in range(len(channel_out)-1)]

        self.residual_mode = residual_mode
        self.kernel_size = kernel_size
        self.layers = torch.nn.ModuleList([])
        c_in = channel_in
        #for c_out, c_extra in zip(channel_out[:-1], [0]+residual_channel):
        for c_out, c_extra in zip(channel_out, [0]+residual_channel):
            self.layers.append(
                UpsamplingBlock(c_in + c_extra,
                                c_out,
                                kernel_size,
                                upsample_rate,
                                upsample_mode)
            )
            c_in = c_out
        """
        self.layers.append(torch.nn.Conv1d(
            c_in + residual_channel[-1],
            channel_out[-1],
            kernel_size=1,
        ))
        """

    def forward(self,
                x : torch.Tensor,
                r : tp.List[torch.Tensor]=None,) -> torch.Tensor:
        """
        Perform decoding

        Parameter
        ---------
        x : torch.Tensor
        r : tp.List[torch.Tensor]

        Returns
        -------
        torch.Tensor : returns_each=False
        List[torch.Tensor] : returns_each=True
        """

        if self.residual_mode == 'none':
            r = [None] * (len(self.layers) - 1)
            merge_in = lambda y, s: y
        elif self.residual_mode == 'concat':
            assert len(r) == len(self.layers) - 1
            merge_in = lambda y, s: \
                torch.cat((y, match_length(s, y.shape[-1])), dim=1)
        elif self.residual_mode == 'add':
            assert len(r) == len(self.layers) - 1
            merge_in = lambda y, s: \
                y + match_length(s, y.shape[-1])

        out = x
        for l, s in zip(self.layers, [None]+r):
            out = l(merge_in(out, s) if s is not None else out)
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
        #for l in self.layers[:-1]:
        for l in self.layers:
            l_in = l.forward_length(l_in)
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
        #for l in self.layers[-2::-1]:
        for l in self.layers[::-1]:
            l_out = l.reverse_length(l_out)
        return l_out

class WaveUNet(torch.nn.Module):
    """
    WaveUNet module
    """
    def __init__(self,
                 channel_in : int,
                 channel_out : int,
                 channel_enc_dec : tp.List[int],
                 channel_mid : tp.List[int],
                 channel_out_m1 : int,
                 kernel_size_d : int,
                 kernel_size_u : int,
                 kernel_size_m : int,
                 kernel_size_o : int,
                 downsample_rate : int=2,
                 upsample_mode : str='nearest',
                 residual_mode='add',) -> None:
        """
        Parameter
        ---------
        channel_in : int
        channel_out : int
        channel_enc_dec : tp.List[int]
        channel_mid : tp.List[int]
        channel_out_m1 : int
        kernel_size_d : int
        kernel_size_u : int
        kernel_size_m : int
        kernel_size_o : int
        downsample_rate : int
        residual_mode
            'none', '
        """
        super(WaveUNet, self).__init__()
        assert residual_mode in ('none', 'add', 'concat')

        self.encoder = Encoder(channel_in=channel_in,
                               channel_out=channel_enc_dec,
                               kernel_size=kernel_size_d,
                               downsample_rate=downsample_rate)

        self.middle_conv = torch.nn.ModuleList()
        for c_in, c_out in zip([channel_enc_dec[-1], *channel_mid],
                               channel_mid):
            self.middle_conv.append(
                torch.nn.Conv1d(c_in, c_out, kernel_size_m))
            init_conv_weight(self.middle_conv[-1])

        decoder_in_channel = channel_mid[-1] if len(channel_mid) else \
            channel_enc_dec[-1]
        decoder_out_channel = [*channel_enc_dec[-2::-1], channel_out_m1]
        self.decoder = Decoder(channel_in=decoder_in_channel,
                               channel_out=decoder_out_channel,
                               kernel_size=kernel_size_u,
                               upsample_rate=downsample_rate,
                               upsample_mode=upsample_mode,
                               residual_mode=residual_mode,
                               residual_channel=channel_enc_dec[-2::-1])

        self.out_conv = torch.nn.Conv1d(channel_out_m1,
                                        channel_out,
                                        kernel_size_o)
        init_conv_weight(self.out_conv)

        self.residual_mode = residual_mode
        self.kernel_size_m = kernel_size_m
        self.kernel_size_o = kernel_size_o
        self.encoder_out_channel = decoder_in_channel

    def forward(self,
                x : torch.Tensor,
                return_embd : bool=False) -> torch.Tensor:
        """
        Parameter
        ---------
        x : torch.Tensor
        return_embd : bool
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        encoder_out = self.encoder(x, return_each=True)

        residual_in, encoder_out = encoder_out[-2::-1], encoder_out[-1]
        embd = encoder_out
        for l in self.middle_conv:
            embd = l(embd)

        if self.residual_mode == 'none' or len(self.middle_conv) == 0:
            decoder_in = embd
        elif self.residual_mode == 'concat':
            decoder_in = torch.cat(
                (embd, match_length(encoder_out, embd.shape[-1])), dim=1)
        elif self.residual_mode == 'add':
            decoder_in = embd + match_length(encoder_out, embd.shape[-1])

        decoder_out = self.decoder(embd, residual_in)
        decoder_out = self.out_conv(decoder_out)
        if return_embd:
            return decoder_out, embd
        else:
            return decoder_out

    def forward_length(self, l_in : int) -> int:
        """
        Parameter
        ---------
        l_in : int
        """
        l_in = self.encoder.forward_length(l_in)
        for _ in self.middle_conv:
            l_in = l_in - self.kernel_size_m + 1
        l_in = self.decoder.forward_length(l_in)
        return l_in - self.kernel_size_o + 1

    def reverse_length(self, l_out : int) -> int:
        """
        Parameter
        ---------
        l_out : int
        """
        l_out = l_out + self.kernel_size_o - 1
        l_out = self.decoder.reverse_length(l_out)
        for _ in self.middle_conv:
            l_out = l_out + self.kernel_size_o - 1
        return self.encoder.reverse_length(l_out)

    def forward_embd_feature(self) -> int:
        return self.encoder_out_channel

    def forward_embd_length(self, l_in : int) -> int:
        l_in = self.encoder.forward_length(l_in)
        for _ in self.middle_conv:
            l_in = l_in - self.kernel_size_m + 1
        return l_in

    def parameter_list(self, base_lr):
        return self.parameters()

class ChimeraWaveUNet(torch.nn.Module):
    """
    WaveUNet module
    """
    def __init__(self,
                 channel_in : int,
                 channel_out : int,
                 channel_enc_dec : tp.List[int],
                 channel_mid : tp.List[int],
                 channel_out_m1 : int,
                 embd_feature : int,
                 embd_dim : int,
                 kernel_size_d : int,
                 kernel_size_u : int,
                 kernel_size_m : int,
                 kernel_size_o : int,
                 downsample_rate : int=2,
                 upsample_mode : str='nearest',
                 residual_mode='add',) -> None:
        """
        Parameter
        ---------
        channel_in : int
        channel_out : int
        channel_enc_dec : tp.List[int]
        channel_mid : tp.List[int]
        channel_out_m1 : int
        embd_feature : int,
        embd_dim : int,
        kernel_size_d : int
        kernel_size_u : int
        kernel_size_m : int
        kernel_size_o : int
        downsample_rate : int
        residual_mode
            'none', '
        """
        super(ChimeraWaveUNet, self).__init__()

        self.waveunet = WaveUNet(channel_in=channel_in,
                                 channel_out=channel_out,
                                 channel_enc_dec=channel_enc_dec,
                                 channel_mid=channel_mid,
                                 channel_out_m1=channel_out_m1,
                                 kernel_size_d=kernel_size_d,
                                 kernel_size_u=kernel_size_u,
                                 kernel_size_m=kernel_size_m,
                                 kernel_size_o=kernel_size_o,
                                 downsample_rate=downsample_rate,
                                 upsample_mode=upsample_mode,
                                 residual_mode=residual_mode,)

        self.conv = torch.nn.Conv1d(self.waveunet.forward_embd_feature(),
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
        y, embd = self.waveunet(x, return_embd=True)
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
        return self.waveunet.forward_length(l_in)

    def reverse_length(self, l_out : int) -> int:
        """
        Parameter
        ---------
        l_out : int
        """
        return self.waveunet.reverse_length(l_out)

    def forward_embd_feature(self) -> int:
        return self.embd_feature

    def forward_embd_length(self, l_in : int) -> int:
        return self.waveunet.forward_embd_length(l_in)

    def get_core_model(self) -> torch.nn.Module:
        return self.waveunet

    def parameter_list(self, base_lr):
        return self.parameters()

