
import typing as tp
import torch

from .utils import init_lstm_weight
from .commonlayers import (
    STFTLayer,
    ISTFTLayer,
    TrainableSTFTLayer,
    TrainableISTFTLayer,
    CodebookMask,
    SigmoidMask,
)

class BaselineEncoder(torch.nn.Module):
    def __init__(self,
                 in_channel : int,
                 n_fft : int,
                 n_lstm : int,
                 lstm_channel : int,
                 hop_length=None) -> None:
        super().__init__()

        self.stft = TrainableSTFTLayer(n_fft, hop_length)
        self.blstm = torch.nn.LSTM(n_fft * in_channel,
                                   lstm_channel // 2,
                                   n_lstm,
                                   batch_first=True,
                                   bidirectional=True)
        init_lstm_weight(self.blstm, lstm_channel // 2)

        self.in_channel = in_channel
        self.lstm_channel = lstm_channel

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # encode
        X = self.stft(x)
        lstm_out, _ = self.blstm(X.flatten(-4, -2).transpose(-1, -2))
        lstm_out = lstm_out.transpose(-1, -2)
        return lstm_out, X

    def forward_length(self, l_in : int) -> int:
        return self.stft.forward_length(l_in)

    def reverse_length(self, l_out : int) -> int:
        return self.stft.reverse_length(l_out)

    def forward_feature_size(self) -> int:
        return self.lstm_channel

    def parameter_list(self, base_lr):
        return [
            {'params': self.stft.parameters(), 'lr': base_lr * 1e-3},
            {'params': self.blstm.parameters()},
        ]


class BaselineDecoder(torch.nn.Module):
    def __init__(self,
                 in_channel : int,
                 out_channel : int,
                 lstm_channel : int,
                 n_fft : int,
                 magbook_size : int=1,
                 phasebook_size : int=1,
                 mask_each : bool=True,
                 hop_length=None) -> None:
        super().__init__()

        if magbook_size > 1 or phasebook_size > 1:
            assert magbook_size > 1 and phasebook_size > 1

        self.istft = TrainableISTFTLayer(n_fft, hop_length)

        mask_num = in_channel \
            * (out_channel if mask_each else out_channel - 1)
        if magbook_size > 1 and phasebook_size > 1:
            self.mask_module = CodebookMask(
                lstm_channel,
                n_fft,
                mask_num=mask_num,
                magbook_size=magbook_size,
                phasebook_size=phasebook_size,
            )
        else:
            self.mask_module = SigmoidMask(
                lstm_channel,
                n_fft,
                mask_num=mask_num
            )

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.lstm_channel = lstm_channel
        self.mask_each = mask_each

    def forward(self, lstm_out : torch.Tensor, X : torch.Tensor) \
        -> torch.Tensor:
        F = X.shape[-2] // 2
        # infer mask
        masks = self.mask_module(lstm_out)

        # apply mask
        X = X.unsqueeze(-5)
        masks = masks.unflatten(
            -4,
            (self.out_channel if self.mask_each else self.out_channel - 1,
             self.in_channel)

        )
        m_re, m_im = [m.squeeze(-3) for m in masks.split(1, dim=-3)]
        b_re, b_im = [m.squeeze(-3) for m in X.split(1, dim=-3)]
        masked_features = torch.stack((
            m_re * b_re - m_im * b_im,
            m_re * b_im + m_im * b_re,
        ), dim=-3)

        # decode
        if not self.mask_each:
            other_feature = X.squeeze(-5) \
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
            {'params': self.mask_module.parameters()},
            {'params': self.istft.parameters(), 'lr': base_lr * 1e-3},
        ]
