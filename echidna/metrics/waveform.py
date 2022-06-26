
import torch

from .loss import Loss

class _WaveformLoss(Loss):
    @property
    def domains(self):
        return ('waves',)

class NegativeSDRLoss(_WaveformLoss):
    def forward_no_reduction(self, s_pred, s_true):
        raw = - source_to_distortion_ratio(
            s_pred,
            s_true,
            scale_invariant=False,
            scale_dependent=False,
        )
        return raw

class NegativeSISDRLoss(_WaveformLoss):
    def forward_no_reduction(self, s_pred, s_true):
        raw = - source_to_distortion_ratio(
            s_pred,
            s_true,
            scale_invariant=True,
            scale_dependent=False,
        )
        return raw

class NegativeSDSDRLoss(_WaveformLoss):
    def forward_no_reduction(self, s_pred, s_true):
        raw = - source_to_distortion_ratio(
            s_pred,
            s_true,
            scale_invariant=False,
            scale_dependent=True,
        )
        return raw

class L1WaveformLoss(_WaveformLoss):
    def forward_no_reduction(self, s_pred, s_true):
        return torch.mean(torch.abs(s_pred - s_true), dim=(1, 2))

class L2WaveformLoss(_WaveformLoss):
    def forward_no_reduction(self, s_pred, s_true):
        return torch.mean((s_pred - s_true) ** 2, dim=(1, 2))

def source_to_distortion_ratio(s_pred : torch.Tensor,
                               s_true : torch.Tensor,
                               scale_invariant : bool=False,
                               scale_dependent : bool=False) -> torch.Tensor:
    """
    loss function for source to distortion ratio
    s_pred: (batch_size, *, waveform_length)
    s_true: (batch_size, *, waveform_length)
    """
    if scale_invariant and scale_dependent:
        raise ValueError(
            'at most one of scale_invariant and scale_dependent can be set')

    epsilon = max(torch.finfo(s_true.dtype).eps,
                  torch.finfo(s_pred.dtype).eps)

    scale = torch.sum(s_pred * s_true, dim=-1) \
        / torch.sum((s_true ** 2), dim=-1).clamp(min=epsilon)

    if scale_invariant:
        s_true = s_true * scale.unsqueeze(-1)

    sdr = 10 * torch.log10(
        torch.sum(s_true**2, axis=-1)
        / torch.sum(((s_true - s_pred) ** 2), axis=-1).clamp(min=epsilon)
        + epsilon
    )

    if scale_dependent:
        sd_sdr = sdr + 10 * torch.log10(scale ** 2 + epsilon)
        sdr = torch.where(sd_sdr < sdr, sd_sdr, sdr)

    return sdr.mean(dim=1)

