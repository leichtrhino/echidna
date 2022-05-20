
import typing as tp
from math import pi, log2, ceil, floor, sqrt, cos, sin
import torch

def _generate_dft_matrix(n_fft):
    phi = 2*pi*torch.arange(1, n_fft + 1, dtype=torch.float) / n_fft
    basis = torch.arange(n_fft // 2, dtype=torch.float).unsqueeze(-1)
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
                                    n_fft,
                                    n_fft,
                                    stride=hop_length,
                                    padding=hop_length * 2,
                                    padding_mode='reflect',
                                    bias=False,)

        weight = torch.sqrt(torch.hann_window(n_fft)) \
            * _generate_dft_matrix(n_fft)
        with torch.no_grad():
            self.conv.weight.copy_(weight.unsqueeze(1))

    def forward(self, x):
        """
        input: (batch_size, n_channels, waveform_length)
        output: (batch_size, n_channels, 2, (n_fft//2+1), time)
        """
        x_shape = x.shape[:-1]
        return self.conv(x.flatten(0, -2).unsqueeze(1))\
                   .unflatten(1, (2, self.n_fft // 2))\
                   .unflatten(0, x_shape)

    def forward_length(self, l_in : int) -> int:
        return (l_in + 2 * self.hop_length*2 - self.n_fft) \
            // self.hop_length + 1

    def reverse_length(self, l_out : int) -> int:
        return self.hop_length * (l_out - 1) \
            - 2 * self.hop_length*2 \
            + self.n_fft

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
        self.conv = torch.nn.ConvTranspose1d(n_fft,
                                             1,
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
        input: (batch_size, n_channels, 2, (n_fft//2+1), time)
        output: (batch_size, n_channels, waveform_length)
        """
        x_shape = x.shape[:-3]
        return self.conv(x.flatten(0, -4).flatten(1, 2))\
                   .squeeze(-2)\
                   .unflatten(0, x_shape) / self.n_fft

    def forward_length(self, l_in : int) -> int:
        return self.hop_length * (l_in - 1) - 2 * self.hop_length*2 + self.n_fft

    def reverse_length(self, l_out : int) -> int:
        return ceil((l_out + 2 * self.hop_length*2 - self.n_fft)
                    / self.hop_length) + 1


class StftLayer(torch.nn.Module):
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
        super(StftLayer, self).__init__()
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

class IstftLayer(torch.nn.Module):
    def __init__(self,
                 n_fft : int,
                 hop_length : int=None,) -> None:
        super(IstftLayer, self).__init__()

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


class RestrictedBiLSTM(torch.nn.Module):
    """
    """
    def __init__(self,
                 input_size : int,
                 output_size : int,
                 freq_dim : int,
                 hidden_size : int,
                 num_layers : int,
                 span : int,
                 stride : int) -> None:
        super(RestrictedBiLSTM, self).__init__()
        assert span >= stride

        self.blstm = torch.nn.LSTM(input_size * freq_dim,
                                   hidden_size,
                                   num_layers,
                                   batch_first=True,
                                   bidirectional=True)
        self.linear = torch.nn.Conv1d(2 * hidden_size,
                                      output_size * freq_dim,
                                      kernel_size=1)
        _init_conv_weight(self.linear)

        self.freq_dim = freq_dim
        self.span = span
        self.stride = stride

    def forward(self, x):
        orig_length = x.shape[-1]
        n_frames = max(1, ceil((orig_length - self.span) / self.stride) + 1)
        padded_length = (n_frames - 1) * self.stride + self.span
        pad_left = (padded_length - orig_length) // 2

        is_temporal = len(x.shape) == 3
        if is_temporal:
            x = x.unsqueeze(2)
            assert self.freq_dim == 1
        else:
            assert self.freq_dim == x.shape[2]

        # B, C, F, T
        # -> B, C*F*self.span, n_frames
        # -> B, C*F, self.span, n_frames
        # -> B, n_frames, C*F, self.span
        # -> B*n_frames, C*F, self.span
        unfolded = torch.nn.functional.unfold(
            x,
            kernel_size=(x.shape[2], self.span),
            padding=(0, pad_left+1),
            stride=(1, self.stride)
        ).unflatten(1, (x.shape[1]*x.shape[2], self.span))\
                                      .permute(0, 3, 1, 2)\
                                      .flatten(0, 1)

        # B*n_frames, C*F, self.span
        # -> B*n_frames, self.span, C*F
        # -> B*n_frames, self.span, D*2
        blstm_in = unfolded.transpose(-2, -1)
        blstm_out = self.blstm(blstm_in)[0]

        # B*n_frames, self.span, D*2
        # -> B*n_frames, D*2, self.span
        # -> B*n_frames, C_out*F, self.span
        linear_in = blstm_out.transpose(-2, -1)
        linear_out = self.linear(linear_in)

        # B*n_frames, C_out*F, self.span
        # -> B, n_frames, C_out*F, self.span
        # -> B, n_frames, C_out, F, self.span
        # -> B, C_out, F, self.span, n_frames
        # -> B, C_out*F*self.span, n_frames
        # -> B, C_out, F, L_out
        fold_in = linear_out.unflatten(0, (x.shape[0], -1))\
                            .unflatten(2, (-1, x.shape[2]))\
                            .permute(0, 2, 3, 4, 1)\
                            .flatten(1, 3)
        folded = torch.nn.functional.fold(
            fold_in,
            output_size=(x.shape[2], orig_length),
            kernel_size=(x.shape[2], self.span),
            padding=(0, pad_left+1),
            stride=(1, self.stride)
        )

        if is_temporal:
            folded = folded.squeeze(2)

        return folded

class LocalAttention(torch.nn.Module):
    """
    """

    def __init__(self,
                 embd_dim : int,
                 freq_dim : int,
                 num_heads : int,
                 num_penalize : int):
        """
        """
        super(LocalAttention, self).__init__()

        self.query_conv = torch.nn.Conv1d(embd_dim * freq_dim, embd_dim, 1)
        self.key_conv = torch.nn.Conv1d(embd_dim * freq_dim, embd_dim, 1)
        self.value_conv = torch.nn.Conv1d(embd_dim * freq_dim, embd_dim, 1)

        self.penalize_conv = torch.nn.Conv1d(embd_dim * freq_dim,
                                             num_heads * num_penalize,
                                             1)
        self.penalize_factors = torch.nn.Parameter(
            torch.arange(1, num_penalize+1, dtype=torch.float32),
            requires_grad=False
        )

        self.proj_conv = torch.nn.Conv1d(embd_dim,
                                         embd_dim * freq_dim,
                                         1)

        _init_conv_weight(self.query_conv)
        _init_conv_weight(self.key_conv)
        _init_conv_weight(self.value_conv)
        _init_conv_weight(self.penalize_conv)
        self.penalize_conv.weight.data *= 1e-3
        _init_conv_weight(self.proj_conv)

        self.embd_dim = embd_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.num_penalize = num_penalize

    def forward(self, x):
        is_spectral = len(x.shape) == 4
        if is_spectral:
            # reshape B, C, F, T into B, C*F, T
            assert x.shape[2] == self.freq_dim
            x = x.flatten(1, 2)
        else:
            assert self.freq_dim == 1

        B, C, T = x.shape

        # prepare penalize term
        pos = torch.arange(T)
        diff_mat = torch.abs(pos.unsqueeze(0) - pos.unsqueeze(-1))\
                        .to(x.device, dtype=x.dtype)
        penalize_base = self.penalize_conv(x).view(B*self.num_heads, -1, T) \
                                             .sigmoid()
        penalize_term = torch.sum(
            self.penalize_factors[None, :, None] * penalize_base,
            dim=1
        ).unsqueeze(2) \
        * diff_mat.unsqueeze(0)

        # calculate attention
        # calculate query and key embedding
        # NOTE: these are output of conv, and shape of (B, C, T)
        query = self.query_conv(x).view(B*self.num_heads, -1, T)\
                                  .transpose(1, 2)
        key = self.key_conv(x).view(B*self.num_heads, -1, T)\
                              .transpose(1, 2)
        values = self.value_conv(x).view(B*self.num_heads, -1, T)\
                                   .transpose(1, 2)
        weight = torch.softmax(
            query.bmm(key.transpose(1, 2)) - penalize_term,
            dim=-1
        )
        attention = weight.bmm(values)

        proj = self.proj_conv(attention.transpose(1, 2).reshape(B, -1, T))

        if is_spectral:
            # reshape B*F, C, T into B, C, F, T
            proj = proj.unflatten(1, (self.embd_dim, self.freq_dim))

        return proj
    pass

class CompressedResidualBlock(torch.nn.Module):
    def __init__(self,
                 # io channels
                 in_channel : int,
                 out_channel : int,
                 freq_dim : int,
                 # bottleneck parameters
                 bottleneck_channel : int,
                 kernel_size : tp.Union[int, tp.Tuple[int]],
                 dilation : tp.Union[int, tp.Tuple[int]],
                 # lstm parameters
                 lstm_layers : int=None,
                 lstm_span : int=None,
                 lstm_stride : int=None,
                 # local attention parameters
                 local_attention_heads : int=None,
                 local_attention_penalize : int=None,
                 # scale parameters
                 init_scale : float=1e-3) -> None:
        """
        """
        super(CompressedResidualBlock, self).__init__()

        assert type(kernel_size) in (int, list, tuple)
        self.is_temporal = type(kernel_size) == int

        if self.is_temporal:
            assert kernel_size % 2 == 1
            assert type(dilation) == int
            assert freq_dim == 1
        else:
            assert len(kernel_size) == 2
            assert all(k % 2 == 1 for k in kernel_size)
            assert type(dilation) in (list, tuple)
            assert len(dilation) == len(kernel_size)

        conv_class = torch.nn.Conv1d if self.is_temporal \
            else torch.nn.Conv2d
        padding = dilation * (kernel_size - 1) // 2 if self.is_temporal \
            else [d * (k - 1) // 2 for k, d in zip(kernel_size, dilation)]

        self.enc_conv = conv_class(in_channel,
                                   bottleneck_channel,
                                   kernel_size,
                                   dilation=dilation,
                                   padding=padding,
                                   bias=False)
        self.enc_norm = torch.nn.GroupNorm(1, bottleneck_channel)
        self.enc_activation = torch.nn.GELU()
        _init_conv_weight(self.enc_conv)

        self.with_lstm = lstm_layers is not None
        if self.with_lstm:
            self.lstm = RestrictedBiLSTM(
                input_size=bottleneck_channel,
                output_size=bottleneck_channel,
                freq_dim=freq_dim,
                hidden_size=bottleneck_channel,
                num_layers=lstm_layers,
                span=lstm_span,
                stride=lstm_stride,
            )

        self.with_attention = local_attention_heads is not None
        if self.with_attention:
            assert bottleneck_channel % local_attention_heads == 0
            self.attention = LocalAttention(
                embd_dim=bottleneck_channel,
                freq_dim=freq_dim,
                num_heads=local_attention_heads,
                num_penalize=local_attention_penalize
            )

        self.dec_conv = conv_class(bottleneck_channel,
                                   out_channel * 2,
                                   1,
                                   bias=False)
        self.dec_norm = torch.nn.GroupNorm(1, out_channel * 2)
        self.dec_activation = torch.nn.GLU(dim=1)
        _init_conv_weight(self.dec_conv)

        self.scale = torch.nn.Parameter(
            torch.full((out_channel,), init_scale),
            requires_grad=True
        )


    def forward(self, x):
        assert len(x.shape) == (3 if self.is_temporal else 4)

        enc_out = self.enc_activation(self.enc_norm(self.enc_conv(x)))
        if self.with_lstm:
            enc_out = enc_out + self.lstm(enc_out)
        if self.with_attention:
            enc_out = enc_out + self.attention(enc_out)

        dec_out = self.dec_activation(self.dec_norm(self.dec_conv(enc_out)))

        scale = self.scale[None, :]
        while len(scale.shape) < len(dec_out.shape):
            scale = scale.unsqueeze(-1)

        return scale * dec_out

class BottleneckBlock(torch.nn.Module):
    def __init__(self,
                 # mode selector
                 is_transposed : bool,
                 # io channels
                 in_channel : int,
                 out_channel : int,
                 freq_dim : int,
                 # conv parameters
                 kernel_size : tp.Union[int, tp.Tuple[int]],
                 stride : tp.Union[int, tp.Tuple[int]],
                 norm_groups : int,
                 # compress residual parameters
                 compress_layers : int=0,
                 compress_channel : int=None,
                 compress_kernel_size : tp.Union[int, tp.Tuple[int]]=None,
                 compress_dilation_multiply : tp.Union[int, tp.Tuple[int]]=None,
                 # lstm parameters
                 compress_lstm_layers : int=None,
                 compress_lstm_span : int=None,
                 compress_lstm_stride : int=None,
                 # local attention parameters
                 compress_attention_heads : int=None,
                 compress_attention_penalize : int=None,
                 # scale parameters
                 compress_init_scale : float=1e-3) -> None:
        super(BottleneckBlock, self).__init__()

        assert type(kernel_size) in (int, list, tuple)
        if type(kernel_size) == int:
            assert type(stride) == int
            assert (kernel_size - stride) % 2 == 0

        if type(kernel_size) in (list, tuple):
            assert len(kernel_size) == 2
            assert type(stride) in (list, tuple)
            assert len(stride) == len(kernel_size)
            assert all((k - s) % 2 == 0 for k, s in zip(kernel_size, stride))

        is_temporal = type(kernel_size) == int
        if not is_transposed:
            conv_class = torch.nn.Conv1d if is_temporal \
                else torch.nn.Conv2d
        else:
            conv_class = torch.nn.ConvTranspose1d if is_temporal \
                else torch.nn.ConvTranspose2d
        padding = (kernel_size - stride) // 2 if is_temporal \
            else [(k - s) // 2 for k, s in zip(kernel_size, stride)]

        self.enc_conv = conv_class(in_channel,
                                   out_channel,
                                   kernel_size,
                                   stride=stride,
                                   padding=padding)
        if norm_groups > 0:
            self.enc_norm = torch.nn.GroupNorm(norm_groups, out_channel)
        self.enc_activation = torch.nn.GELU()
        _init_conv_weight(self.enc_conv)

        if compress_layers > 0:
            self.compress_blocks = torch.nn.ModuleList()
            for bi in range(compress_layers):
                compress_dilation = None
                if is_temporal:
                    compress_dilation = compress_dilation_multiply ** bi
                else:
                    compress_dilation = \
                        [d**bi for d in compress_dilation_multiply]
                self.compress_blocks.append(CompressedResidualBlock(
                    in_channel=out_channel,
                    out_channel=out_channel,
                    freq_dim=(freq_dim if is_temporal else
                              freq_dim * stride[0] if is_transposed else
                              freq_dim // stride[0]),
                    bottleneck_channel=compress_channel,
                    kernel_size=compress_kernel_size,
                    dilation=compress_dilation,
                    lstm_layers=compress_lstm_layers,
                    lstm_span=compress_lstm_span,
                    lstm_stride=compress_lstm_stride,
                    local_attention_heads=compress_attention_heads,
                    local_attention_penalize=compress_attention_penalize,
                    init_scale=compress_init_scale,
                ))

        self.dec_conv = conv_class(out_channel,
                                   out_channel * 2,
                                   kernel_size=1,
                                   stride=1)
        if norm_groups > 0:
            self.dec_norm = torch.nn.GroupNorm(norm_groups, out_channel * 2)
        self.dec_activation = torch.nn.GLU(dim=1)
        _init_conv_weight(self.dec_conv)

        self.is_transposed = is_transposed
        self.is_temporal = is_temporal
        self.use_norm = norm_groups > 0
        self.has_compress = compress_layers > 0
        self.temporal_stride = stride if is_temporal else stride[1]

    def forward(self, x):
        assert len(x.shape) == (3 if self.is_temporal else 4)

        enc_out = self.enc_conv(x)
        if self.use_norm:
            enc_out = self.enc_norm(enc_out)
        enc_out = self.enc_activation(enc_out)

        if self.has_compress:
            for b in self.compress_blocks:
                enc_out = enc_out + b(enc_out)

        dec_out = self.dec_conv(enc_out)
        if self.use_norm:
            dec_out = self.dec_norm(dec_out)
        dec_out = self.dec_activation(dec_out)
        return dec_out

    def forward_length(self, l_in):
        if not self.is_transposed:
            return l_in // self.temporal_stride
        else:
            return l_in * self.temporal_stride

    def reverse_length(self, l_out):
        if not self.is_transposed:
            return l_out * self.temporal_stride
        else:
            return ceil(l_out / self.temporal_stride)

# pseudo-proxy class definitions
def TEncoderBlock(in_channel : int,
                  out_channel : int,
                  # conv parameters
                  kernel_size : int,
                  stride : int,
                  norm_groups : int,
                  # compress parameters
                  compress_layers : int,
                  compress_channel : int,
                  compress_kernel_size : int,
                  compress_dilation_multiply : int,
                  # lstm parameters
                  compress_lstm_layers : int=None,
                  compress_lstm_span : int=None,
                  compress_lstm_stride : int=None,
                  # local attention parameters
                  compress_attention_heads : int=None,
                  compress_attention_penalize : int=None,
                  # scale parameters
                  compress_init_scale : float=1e-3):

    return BottleneckBlock(
        is_transposed=False,
        # io channels
        in_channel=in_channel,
        out_channel=out_channel,
        freq_dim=1,
        # conv parameters
        kernel_size=kernel_size,
        stride=stride,
        norm_groups=norm_groups,
        # compress parameters
        compress_layers=compress_layers,
        compress_channel=compress_channel,
        compress_kernel_size=compress_kernel_size,
        compress_dilation_multiply=compress_dilation_multiply,
        # lstm parameters
        compress_lstm_layers=compress_lstm_layers,
        compress_lstm_span=compress_lstm_span,
        compress_lstm_stride=compress_lstm_stride,
        # local attention parameters
        compress_attention_heads=compress_attention_heads,
        compress_attention_penalize=compress_attention_penalize,
        # scale parameters
        compress_init_scale=compress_init_scale,
    )

def TDecoderBlock(in_channel : int,
                  out_channel : int,
                  # conv parameters
                  kernel_size : int,
                  stride : int,
                  norm_groups : int):

    return BottleneckBlock(
        is_transposed=True,
        # io channels
        in_channel=in_channel,
        out_channel=out_channel,
        freq_dim=1,
        # conv parameters
        kernel_size=kernel_size,
        stride=stride,
        norm_groups=norm_groups,
    )

def ZEncoderBlock(in_channel : int,
                  out_channel : int,
                  freq_dim : int,
                  # conv parameters
                  kernel_size : int,
                  stride : int,
                  norm_groups : int,
                  # compress parameters
                  compress_layers : int,
                  compress_channel : int,
                  compress_kernel_size : int,
                  compress_dilation_multiply : int,
                  # lstm parameters
                  compress_lstm_layers : int=None,
                  compress_lstm_span : int=None,
                  compress_lstm_stride : int=None,
                  # local attention parameters
                  compress_attention_heads : int=None,
                  compress_attention_penalize : int=None,
                  # scale parameters
                  compress_init_scale : float=1e-3):

    return BottleneckBlock(
        is_transposed=False,
        # io channels
        in_channel=in_channel,
        out_channel=out_channel,
        freq_dim=freq_dim,
        # conv parameters
        kernel_size=(kernel_size, 1),
        stride=(stride, 1),
        norm_groups=norm_groups,
        # compress parameters
        compress_layers=compress_layers,
        compress_channel=compress_channel,
        compress_kernel_size=(compress_kernel_size, 1),
        compress_dilation_multiply=(compress_dilation_multiply, 1),
        # lstm parameters
        compress_lstm_layers=compress_lstm_layers,
        compress_lstm_span=compress_lstm_span,
        compress_lstm_stride=compress_lstm_stride,
        # local attention parameters
        compress_attention_heads=compress_attention_heads,
        compress_attention_penalize=compress_attention_penalize,
        # scale parameters
        compress_init_scale=compress_init_scale,
    )

def ZDecoderBlock(in_channel : int,
                  out_channel : int,
                  freq_dim : int,
                  # conv parameters
                  kernel_size : int,
                  stride : int,
                  norm_groups : int,):

    return BottleneckBlock(
        is_transposed=True,
        # io channels
        in_channel=in_channel,
        out_channel=out_channel,
        freq_dim=freq_dim,
        # conv parameters
        kernel_size=(kernel_size, 1),
        stride=(stride, 1),
        norm_groups=norm_groups,
    )


class DemucsV3(torch.nn.Module):
    def __init__(self,
                 # architecture parameters
                 in_channel : int,
                 out_channel : int,
                 mid_channels : tp.List[int],
                 # conv parameters
                 kernel_size : tp.Union[int, tp.List[int]],
                 stride : tp.Union[int, tp.List[int]],
                 inner_kernel_size : int,
                 inner_stride : int,
                 # misc. architecture parameters
                 infer_each : bool=True,
                 embedding_layers : int=1,
                 attention_layers : int=2,
                 groupnorm_layers : int=2,
                 groupnorm_groups : int=4,
                 trainable_stft : bool=False,
                 # compress parameters
                 compress_layers : int=2,
                 compress_channel_scale : int=4, # C_out // 4 for each block
                 compress_kernel_size : int=3,
                 compress_dilation_multiply : int=2,
                 # lstm parameters
                 compress_lstm_layers : int=2,
                 compress_lstm_span : int=200,
                 compress_lstm_stride : int=100,
                 # local attention parameters
                 compress_attention_heads : int=4,
                 compress_attention_penalize : int=4,
                 # scale parameters
                 compress_init_scale : float=1e-3) -> None:

        super(DemucsV3, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.embd_channel = mid_channels[-1]
        self.infer_each = infer_each

        if type(kernel_size) == list:
            assert len(kernel_size) == len(mid_channels) - 1
        else:
            kernel_size = [kernel_size] * (len(mid_channels) - 1)

        if type(stride) == list:
            assert len(stride) == len(mid_channels) - 1
        else:
            stride = [stride] * (len(mid_channels) - 1)

        # initialize STFT and ISTFT layer
        # the number of ZEncoder blocks: len(mid_channels) - 1
        # total reduction along freq. axis:
        #     before last ZEncoder block: (stride ** (len(mid_channels) - 2))
        #     last ZEncoder block: kernel_size
        #     -> n_fft = 2 * kernel_size * stride ** (len(mid_channels)-2)
        #     example: 5 ZEncoders with stride=4 and kernel=8, 2 * 8 * 4**4 = 4096
        n_fft = 2 * kernel_size[-1]
        for s in stride[:-1]:
            n_fft *= s
        # total reduction along time axis: stride ** (len(mid_channels)-1)
        # example: 5 TEncoder with stride=4 and kernel=8, 4**5 = 1024
        hop_length = 1
        for s in stride:
            hop_length *= s
        if trainable_stft:
            self.stft = TrainableStftLayer(n_fft, hop_length)
            self.istft = TrainableIstftLayer(n_fft, hop_length)
        else:
            self.stft = StftLayer(n_fft, hop_length)
            self.istft = IstftLayer(n_fft, hop_length)

        # initialize encoder and decoder blocks
        self.tencoder = torch.nn.ModuleList()
        for ei in range(len(mid_channels)-1):
            c_in = in_channel if ei == 0 else mid_channels[ei-1]
            c_out = mid_channels[ei]
            norm_groups = 0 if ei < len(mid_channels) - groupnorm_layers \
                else groupnorm_groups
            opt_params = dict()
            if ei >= len(mid_channels) - attention_layers:
                opt_params.update(dict(
                    compress_lstm_layers=compress_lstm_layers,
                    compress_lstm_span=compress_lstm_span,
                    compress_lstm_stride=compress_lstm_stride,
                    compress_attention_heads=compress_attention_heads,
                    compress_attention_penalize=compress_attention_penalize,
                ))
            self.tencoder.append(TEncoderBlock(
                in_channel=c_in,
                out_channel=c_out,
                kernel_size=kernel_size[ei],
                stride=stride[ei],
                norm_groups=norm_groups,
                compress_layers=compress_layers,
                compress_channel=c_out // compress_channel_scale,
                compress_kernel_size=compress_kernel_size,
                compress_dilation_multiply=compress_dilation_multiply,
                compress_init_scale=compress_init_scale,
                **opt_params,
            ))

        self.zencoder = torch.nn.ModuleList()
        self.freq_embeddings = torch.nn.ParameterList()
        freq_dim = (n_fft // 2)
        for ei in range(len(mid_channels)-1):
            c_in = 2 * in_channel if ei == 0 else mid_channels[ei-1]
            c_out = mid_channels[ei]
            norm_groups = 0 if ei < len(mid_channels) - groupnorm_layers \
                else groupnorm_groups
            opt_params = dict()
            if ei >= len(mid_channels) - attention_layers:
                opt_params.update(dict(
                    compress_lstm_layers=compress_lstm_layers,
                    compress_lstm_span=compress_lstm_span,
                    compress_lstm_stride=compress_lstm_stride,
                    compress_attention_heads=compress_attention_heads,
                    compress_attention_penalize=compress_attention_penalize,
                ))
            self.zencoder.append(ZEncoderBlock(
                in_channel=c_in,
                out_channel=c_out,
                freq_dim=freq_dim,
                kernel_size=kernel_size[ei],
                stride=stride[ei] if ei < len(mid_channels)-2 else kernel_size[ei],
                norm_groups=norm_groups,
                compress_layers=compress_layers,
                compress_channel=c_out // compress_channel_scale,
                compress_kernel_size=compress_kernel_size,
                compress_dilation_multiply=compress_dilation_multiply,
                compress_init_scale=compress_init_scale,
                **opt_params,
            ))
            if ei > 0 and ei <= embedding_layers:
                self.freq_embeddings.append(torch.nn.Parameter(
                    torch.rand(c_in, freq_dim),
                    requires_grad=True
                ))
            freq_dim //= stride[ei]

        # initialize shared last encoder layer
        opt_params = dict()
        if attention_layers > 0:
            opt_params.update(dict(
                compress_lstm_layers=compress_lstm_layers,
                compress_lstm_span=compress_lstm_span,
                compress_lstm_stride=compress_lstm_stride,
                compress_attention_heads=compress_attention_heads,
                compress_attention_penalize=compress_attention_penalize,
            ))
        self.encoder = TEncoderBlock(
            in_channel=mid_channels[-2],
            out_channel=mid_channels[-1],
            kernel_size=inner_kernel_size,
            stride=inner_stride,
            norm_groups=0 if groupnorm_layers <= 0 else groupnorm_groups,
            compress_layers=compress_layers,
            compress_channel=mid_channels[-1] // compress_channel_scale,
            compress_kernel_size=compress_kernel_size,
            compress_dilation_multiply=compress_dilation_multiply,
            **opt_params,
            compress_init_scale=compress_init_scale,
        )

        # initialize shared first decoder layer
        self.decoder = TDecoderBlock(
            in_channel=mid_channels[-1],
            out_channel=mid_channels[-2],
            kernel_size=inner_kernel_size,
            stride=inner_stride,
            norm_groups=0 if groupnorm_layers <= 0 else groupnorm_groups,
        )

        # initialize TDecoder
        self.tdecoder = torch.nn.ModuleList()
        for ei in range(len(mid_channels)-2, -1, -1):
            c_in = mid_channels[ei]
            if ei == 0:
                c_out = (out_channel if infer_each else 1) * in_channel
            else:
                c_out = mid_channels[ei-1]
            norm_groups = 0 if ei < len(mid_channels) - groupnorm_layers \
                else groupnorm_groups
            self.tdecoder.append(TDecoderBlock(
                in_channel=c_in,
                out_channel=c_out,
                kernel_size=kernel_size[ei],
                stride=stride[ei],
                norm_groups=norm_groups,
            ))

        # initialize ZDecoder
        self.zdecoder = torch.nn.ModuleList()
        freq_dim = 1
        for ei in range(len(mid_channels)-2, -1, -1):
            c_in = mid_channels[ei]
            if ei == 0:
                c_out = 2 * (out_channel if infer_each else 1) * in_channel
            else:
                c_out = mid_channels[ei-1]
            norm_groups = 0 if ei < len(mid_channels) - groupnorm_layers \
                else groupnorm_groups
            self.zdecoder.append(ZDecoderBlock(
                in_channel=c_in,
                out_channel=c_out,
                freq_dim=freq_dim,
                kernel_size=kernel_size[ei],
                stride=stride[ei] if ei < len(mid_channels)-2 else kernel_size[ei],
                norm_groups=norm_groups,
            ))
            freq_dim *= stride[ei]

    def forward(self, x, return_embd=False):
        def align_size(y, size):
            if y.shape[-1] > size:
                lp = (y.shape[-1] - size) // 2
                y = y[..., lp:lp+size]
            elif y.shape[-1] < size:
                lp = (size - y.shape[-1]) // 2
                rp = size - y.shape[-1] - lp
                y = torch.cat((
                    torch.zeros(*y.shape[:-1], lp, device=y.device),
                    y,
                    torch.zeros(*y.shape[:-1], rp, device=y.device),
                ), dim=-1)
            return y

        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # find reverse length for padding
        l_in = self._reverse_length(x.shape[-1])
        pad_left = (l_in - x.shape[-1]) // 2
        pad_right = l_in - x.shape[-1] - pad_left
        x_ = torch.nn.ReflectionPad1d((pad_left, pad_right))(x)

        # prepare encodings on time domain
        t_encs = []
        for e in self.tencoder:
            t_encs.append(e(x_ if len(t_encs) == 0 else t_encs[-1]))

        # prepare encodings on freq. domain
        X = self.stft(x_).flatten(1, 2)
        z_encs = []
        freq_embeddings = [None] + list(self.freq_embeddings) \
            + [None] * (len(self.zencoder)-len(self.freq_embeddings)-1)
        for fe, e in zip(freq_embeddings, self.zencoder):
            e_in = X if len(z_encs) == 0 else z_encs[-1]
            if fe is not None:
                e_in = e_in + fe[None, :, :, None]
            z_encs.append(e(e_in))

        # fuse encodings
        enc_size = max(t_encs[-1].shape[-1], z_encs[-1].shape[-1])
        enc = self.encoder(
            align_size(t_encs[-1], enc_size)
            + align_size(z_encs[-1].squeeze(dim=2), enc_size)
        )

        # decode
        dec = self.decoder(enc)

        # decode along time axis
        assert len(t_encs) == len(self.tdecoder)
        t_dec = dec
        for d, skip in zip(self.tdecoder, t_encs[::-1]):
            dec_size = max(t_dec.shape[-1], skip.shape[-1])
            t_dec = d(align_size(t_dec, dec_size) + align_size(skip, dec_size))
        t_dec = t_dec.view(x.shape[0],
                           self.out_channel if self.infer_each else 1,
                           self.in_channel,
                           -1)

        # decode along freq. axis
        assert len(z_encs) == len(self.zdecoder)
        z_dec = dec.unsqueeze(2)
        for d, skip in zip(self.zdecoder, z_encs[::-1]):
            dec_size = max(z_dec.shape[-1], skip.shape[-1])
            z_dec = d(align_size(z_dec, dec_size) + align_size(skip, dec_size))
        z_dec = self.istft(z_dec.unflatten(1, (-1, 2)))
        z_dec = z_dec.view(x.shape[0],
                           self.out_channel if self.infer_each else 1,
                           self.in_channel,
                           -1)

        # fuse waveforms
        waveform = align_size(t_dec, x.shape[-1]) \
            + align_size(z_dec, x.shape[-1])
        if not self.infer_each:
            waveform = torch.cat((waveform, x.unsqueeze(1) - waveform), dim=1)
        waveform = waveform.squeeze(-3).squeeze(-2)

        if return_embd:
            return waveform, enc
        else:
            return waveform

    def forward_length(self, l_in):
        return l_in

    def reverse_length(self, l_out):
        return l_out

    def _reverse_length(self, l_out):
        l_tdecs = []
        for d in self.tdecoder:
            l_tdecs.append(d.reverse_length(
                l_out if len(l_tdecs) == 0 else l_tdecs[-1]))
        l_zdecs = []
        for d in self.zdecoder:
            l_zdecs.append(d.reverse_length(
                self.istft.reverse_length(l_out) if len(l_zdecs) == 0
                else l_zdecs[-1]))
        l_dec = self.decoder.reverse_length(max(l_tdecs[-1], l_zdecs[-1]))
        l_enc = self.encoder.reverse_length(l_dec)
        l_tenc = l_enc
        for e, l in zip(self.tencoder, l_tdecs[::-1]):
            l_tenc = e.reverse_length(max(l_tenc, l))
        l_zenc = l_enc
        for e, l in zip(self.zencoder, l_zdecs[::-1]):
            l_zenc = e.reverse_length(max(l_zenc, l))
        l_zenc = self.stft.reverse_length(l_zenc)
        return max(l_tenc, l_zenc)

    def forward_embd_feature(self) -> int:
        return self.embd_channel

    def forward_embd_length(self, l_in : int) -> int:
        l_in = self._reverse_length(l_in)
        tenc_in = l_in
        for e in self.tencoder:
            tenc_in = e.forward_length(tenc_in)
        zenc_in = self.stft.forward_length(l_in)
        enc_in = max(tenc_in, zenc_in)
        return self.encoder.forward_length(enc_in)

    def parameter_list(self, base_lr):
        return [
            {'params': self.stft.parameters(), 'lr': base_lr * 1e-3},
            {'params': self.tencoder.parameters()},
            {'params': self.zencoder.parameters()},
            {'params': self.freq_embeddings.parameters()},
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()},
            {'params': self.tdecoder.parameters()},
            {'params': self.zdecoder.parameters()},
            {'params': self.istft.parameters(), 'lr': base_lr * 1e-3},
        ]

class ChimeraDemucs(torch.nn.Module):
    """
    ChimeraDemucs module
    """
    def __init__(self,
                 # architecture parameters
                 in_channel : int,
                 out_channel : int,
                 mid_channels : tp.List[int],
                 # conv parameters
                 kernel_size : int,
                 stride : int,
                 inner_kernel_size : int,
                 inner_stride : int,
                 # deepclustering parameters
                 dc_embd_feature : int,
                 dc_embd_dim : int,
                 # misc. architecture parameters
                 infer_each : bool=True,
                 embedding_layers : int=1,
                 attention_layers : int=2,
                 groupnorm_layers : int=2,
                 groupnorm_groups : int=4,
                 trainable_stft : bool=False,
                 # compress parameters
                 compress_layers : int=2,
                 compress_channel_scale : int=4, # C_out // 4 for each block
                 compress_kernel_size : int=3,
                 compress_dilation_multiply : int=2,
                 # lstm parameters
                 compress_lstm_layers : int=2,
                 compress_lstm_span : int=200,
                 compress_lstm_stride : int=100,
                 # local attention parameters
                 compress_attention_heads : int=4,
                 compress_attention_penalize : int=4,
                 # scale parameters
                 compress_init_scale : float=1e-3) -> None:

        """
        Parameter
        ---------

        """
        super(ChimeraDemucs, self).__init__()

        self.demucs = DemucsV3(
            # architecture parameters
            in_channel=in_channel,
            out_channel=out_channel,
            mid_channels=mid_channels,
            # conv parameters
            kernel_size=kernel_size,
            stride=stride,
            inner_kernel_size=inner_kernel_size,
            inner_stride=inner_stride,
            # misc. architecture parameters
            infer_each=infer_each,
            embedding_layers=embedding_layers,
            attention_layers=attention_layers,
            groupnorm_layers=groupnorm_layers,
            groupnorm_groups=groupnorm_groups,
            trainable_stft=trainable_stft,
            # compress parameters
            compress_layers=compress_layers,
            compress_channel_scale=compress_channel_scale,
            compress_kernel_size=compress_kernel_size,
            compress_dilation_multiply=compress_dilation_multiply,
            # lstm parameters
            compress_lstm_layers=compress_lstm_layers,
            compress_lstm_span=compress_lstm_span,
            compress_lstm_stride=compress_lstm_stride,
            # local attention parameters
            compress_attention_heads=compress_attention_heads,
            compress_attention_penalize=compress_attention_penalize,
            # scale parameters
            compress_init_scale=compress_init_scale,
        )
        self.conv = torch.nn.Conv1d(self.demucs.forward_embd_feature(),
                                    dc_embd_feature * dc_embd_dim,
                                    1)
        _init_conv_weight(self.conv)

        self.embd_feature = dc_embd_feature
        self.embd_dim = dc_embd_dim

    def forward(self,
                x : torch.Tensor) -> torch.Tensor:
        """
        Parameter
        ---------
        x : torch.Tensor
        """
        y, embd = self.demucs(x, return_embd=True)
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
        return self.demucs.forward_length(l_in)

    def reverse_length(self, l_out : int) -> int:
        """
        Parameter
        ---------
        l_out : int
        """
        return self.demucs.reverse_length(l_out)

    def forward_embd_feature(self) -> int:
        return self.embd_feature

    def forward_embd_length(self, l_in : int) -> int:
        return self.demucs.forward_embd_length(l_in)

    def get_core_model(self) -> torch.nn.Module:
        return self.demucs

    def parameter_list(self, base_lr):
        return self.demucs.parameter_list(base_lr)\
            + [{'params': self.conv.parameters()}]

