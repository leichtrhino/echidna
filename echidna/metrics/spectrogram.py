
import math
import torch

from .loss import Loss, register_loss_class

class _SpectrogramLoss(Loss):
    def __init__(self,
                 fft_spec=[(512, 128), (1024, 256), (2048, 512)],
                 reduction='mean'):
        super().__init__(reduction)
        if not fft_spec:
            raise ValueError('empty fft_spec')
        self.fft_spec = fft_spec

    @property
    def domains(self):
        return ('waves',)

    def to_dict_args(self):
        return {
            'reduction': self.reduction,
            'fft_spec': self.fft_spec,
        }

    @classmethod
    def from_dict_args(cls, d : dict):
        return cls(
            fft_spec=d.get('fft_spec', [(512, 128), (1024, 256), (2048, 512)]),
            reduction=d['reduction']
        )

class L1SpectrogramLoss(_SpectrogramLoss):
    def forward_no_reduction(self,
                             s_pred : torch.Tensor,
                             s_true: torch.Tensor):
        raw = multiscale_spectrogram_loss(
            s_pred,
            s_true,
            self.fft_spec,
            spectral_convergence_weight=0.0,
            spectral_magnitude_weight=1.0,
            spectral_magnitude_norm=1,
            spectral_magnitude_log=False)
        return raw

class L2SpectrogramLoss(_SpectrogramLoss):
    def forward_no_reduction(self,
                             s_pred : torch.Tensor,
                             s_true: torch.Tensor):
        raw = multiscale_spectrogram_loss(
            s_pred,
            s_true,
            self.fft_spec,
            spectral_convergence_weight=0.0,
            spectral_magnitude_weight=1.0,
            spectral_magnitude_norm=2,
            spectral_magnitude_log=False)
        return raw

class SpectrogramConvergenceLoss(_SpectrogramLoss):
    def forward_no_reduction(self,
                             s_pred : torch.Tensor,
                             s_true: torch.Tensor):
        raw = multiscale_spectrogram_loss(
            s_pred,
            s_true,
            self.fft_spec,
            spectral_convergence_weight=1.0,
            spectral_magnitude_weight=0.0,
            spectral_magnitude_norm=1,
            spectral_magnitude_log=False
        )
        return raw

class SpectrogramLoss(Loss):
    def __init__(self,
                 fft_spec=[(512, 128), (1024, 256), (2048, 512)],
                 spectral_convergence_weight : float=1.0,
                 spectral_magnitude_weight : float=1.0,
                 spectral_magnitude_norm : int=1,
                 spectral_magnitude_log : bool=True,
                 reduction='mean'):
        super().__init__(reduction)
        if not fft_spec:
            raise ValueError('empty fft_spec')
        self.fft_spec = fft_spec
        self.spectral_convergence_weight = spectral_convergence_weight
        self.spectral_magnitude_weight = spectral_magnitude_weight
        self.spectral_magnitude_norm = spectral_magnitude_norm
        self.spectral_magnitude_log = spectral_magnitude_log

    @property
    def domains(self):
        return ('waves',)

    def to_dict_args(self):
        return {
            'reduction': self.reduction,
            'fft_spec': self.fft_spec,
            'spectral_convergence_weight': self.spectral_convergence_weight,
            'spectral_magnitude_weight': self.spectral_magnitude_weight,
            'spectral_magnitude_norm': self.spectral_magnitude_norm,
            'spectral_magnitude_log': self.spectral_magnitude_log,
        }

    @classmethod
    def from_dict_args(cls, d : dict):
        return cls(
            fft_spec=d.get(
                'fft_spec', [(512, 128), (1024, 256), (2048, 512)]),
            spectral_convergence_weight=d.get(
                'spectral_convergence_weight', 1.0),
            spectral_magnitude_weight=d.get(
                'spectral_magnitude_weight', 1.0),
            spectral_magnitude_norm=d.get(
                'spectral_magnitude_norm', 1),
            spectral_magnitude_log=d.get(
                'spectral_magnitude_log', True),
            reduction=d.get('reduction', 'mean'),
        )

    def forward_no_reduction(self,
                             s_pred : torch.Tensor,
                             s_true: torch.Tensor):
        raw = multiscale_spectrogram_loss(
            s_pred,
            s_true,
            self.fft_spec,
            self.spectral_convergence_weight,
            self.spectral_magnitude_weight,
            self.spectral_magnitude_norm,
            self.spectral_magnitude_log,
        )
        return raw

register_loss_class('l1_spectrogram', L1SpectrogramLoss)
register_loss_class('l2_spectrogram', L2SpectrogramLoss)
register_loss_class('spectrogram_convergence', SpectrogramConvergenceLoss)
register_loss_class('spectrogram', SpectrogramLoss)


def multiscale_spectrogram_loss(
        s_pred : torch.Tensor,
        s_true : torch.Tensor,
        fft_spec : list=[(512, 128), (1024, 256), (2048, 512)],
        spectral_convergence_weight : float=1.0,
        spectral_magnitude_weight : float=1.0,
        spectral_magnitude_norm : int=1,
        spectral_magnitude_log : bool=True,
) -> torch.Tensor:
    """
    loss function for multiscale spectrogram
    s_pred: (batch_size, *, waveform_length)
    s_true: (batch_size, *, waveform_length)
    """
    if not fft_spec:
        raise ValueError('empty fft_spec')

    epsilon = math.sqrt(max(torch.finfo(s_true.dtype).eps,
                            torch.finfo(s_pred.dtype).eps))
    loss = 0
    for n_fft, hop_length in fft_spec:
        window = torch.hann_window(n_fft).to(s_pred.device)
        S_pred = torch.stft(
            s_pred.flatten(0, 1), n_fft, hop_length,
            window=window, return_complex=True
        ).abs().clamp(min=epsilon).unflatten(0, s_pred.shape[:2])
        S_true = torch.stft(
            s_true.flatten(0, 1), n_fft, hop_length,
            window=window, return_complex=True
        ).abs().clamp(min=epsilon).unflatten(0, s_true.shape[:2])

        L_sc = torch.sqrt(torch.sum((S_pred-S_true)**2, dim=(-2, -1))) \
            / torch.sqrt(torch.sum(S_true**2, dim=(-2, -1)))

        if spectral_magnitude_log:
            spec_diff = torch.log(S_pred) - torch.log(S_true)
        else:
            spec_diff = S_pred - S_true
        L_mag = torch.norm(torch.abs(spec_diff).clamp(min=epsilon),
                           p=spectral_magnitude_norm,
                           dim=(-2, -1)) \
                           / (S_pred.shape[-2] * S_pred.shape[-1])

        loss += spectral_convergence_weight * L_sc \
            + spectral_magnitude_weight * L_mag

    return loss.mean(dim=1) / len(fft_spec)

