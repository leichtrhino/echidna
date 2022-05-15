
import typing as tp
import itertools
import math
import torch

def permutation_invariant(loss_function : callable,
                          aggregate_perm_fn : callable=min,
                          aggregate_loss_fn : tp.Union[str, callable]='mean'
                          ) -> callable:
    def _loss_function(*args, **kwargs):
        losses = []
        for a in tuple(zip(*args)):
            l_p = []
            for p in itertools.permutations(range(len(a[0]))):
                l_p.append(
                    loss_function(a[0][p, ...], *a[1:], **kwargs))
            losses.append(aggregate_perm_fn(l_p))

        if type(aggregate_loss_fn) == str and aggregate_loss_fn == 'mean':
            return sum(losses) / len(losses)
        elif type(aggregate_loss_fn) == str and aggregate_loss_fn == 'sum':
            return sum(losses)
        elif callable(aggregate_loss_fn):
            return aggregate_loss_fn(losses)

    return _loss_function

def approximation_loss(s_pred : torch.Tensor,
                       s_true : torch.Tensor,
                       norm : int=1) -> torch.Tensor:
    """
    loss function for wave/spectrogram approximation
    s_pred: (batch_size, *, waveform_length)
        or (batch_size, *, freq_bin, time_stretch)
    s_true: (batch_size, *, waveform_length)
        or (batch_size, *, freq_bin, time_stretch)
    """
    return torch.sum(torch.abs(s_pred - s_true) ** norm) \
        / s_pred.numel()

def deep_clustering_loss(embd : torch.Tensor,
                         label : torch.Tensor,
                         weight : torch.Tensor=None,
                         mode : str='deep-clustering'):
    """
    loss functions for deep clustering
    embd: (batch_size, *, embd_dim)
    label: (batch_size, *, label_dim)
    weight: (batch_size, *)
    """

    if not mode in ('deep-clustering',
                    'deep-lda',
                    'graph-laplacian-distance',
                    'whitened-kmeans'):
        raise ValueError(f'invalid mode {mode}')

    if len(embd.shape) > 3:
        embd = embd.flatten(1, -2)
    if len(label.shape) > 3:
        label = label.flatten(1, -2)
    if weight is not None and len(weight.shape) > 2:
        weight = weight.flatten(1, -1)

    if type(weight) == torch.Tensor:
        weight = torch.sqrt(weight).unsqueeze(-1)
        embd = embd * weight
        label = label * weight

    batch_size, _, C = label.shape
    _, _, D = embd.shape

    if mode == 'deep-clustering':
        return (torch.sum(embd.transpose(1, 2).bmm(embd) ** 2) \
                + torch.sum(label.transpose(1, 2).bmm(label) ** 2) \
                - 2 * torch.sum(embd.transpose(1, 2).bmm(label) ** 2)) \
                / embd.shape[0]

    elif mode == 'deep-lda':
        YtV = label.transpose(1, 2).bmm(embd)
        YtY = label.transpose(1, 2).bmm(label) \
            + torch.eye(C, device=label.device)
        YtYu = torch.linalg.cholesky(YtY)
        return torch.sum(
            (embd - label.bmm(YtYu.cholesky_inverse().bmm(YtV))) ** 2) \
            / torch.sum((embd - embd.mean(dim=-2, keepdims=True)) ** 2) \
            / embd.shape[0]

    elif mode == 'graph-laplacian-distance':
        D_V = torch.bmm(
            embd,
            torch.sum(embd.transpose(1, 2), dim=2, keepdims=True)
        ).clamp(min=1e-24)
        D_Y = torch.bmm(
            label,
            torch.sum(label.transpose(1, 2), dim=2, keepdims=True)
        ).clamp(min=1e-24)
        return torch.sum(
            torch.bmm(embd.transpose(1, 2), D_V ** -1 * embd) ** 2) \
            + C * batch_size \
            - 2 * torch.sum(
                torch.bmm(
                    (embd * torch.sqrt(D_V) ** -1).transpose(1, 2),
                    torch.sqrt(D_Y) ** -1 * label
                ) ** 2
            ) / batch_size

    elif mode == 'whitened-kmeans':
        VtV = embd.transpose(1, 2).bmm(embd) \
            + torch.eye(D, device=embd.device)
        VtVu = torch.linalg.cholesky(VtV)
        VtY = embd.transpose(1, 2).bmm(label)
        YtY = label.transpose(1, 2).bmm(label) \
            + torch.eye(C, device=label.device)
        YtYu = torch.linalg.cholesky(YtY)
        return (D * batch_size \
            - torch.trace(torch.sum(
                VtVu.cholesky_inverse()
                .bmm(VtY)
                .bmm(YtYu.cholesky_inverse())
                .bmm(VtY.transpose(1, 2)),
                dim=0
            ))) / batch_size

def multiscale_spectrogram_loss(
        s_pred : torch.Tensor, s_true : torch.Tensor,
        fft_spec : tp.List[tp.Tuple[int, int]]=[(512, 128), (1024, 256), (2048, 512)],
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
    epsilon = math.sqrt(max(torch.finfo(s_true.dtype).eps,
                            torch.finfo(s_pred.dtype).eps))
    loss = 0
    for n_fft, hop_length in fft_spec:
        window = torch.hann_window(n_fft).to(s_pred.device)
        S_pred = torch.stft(
            s_pred.flatten(0, 1), n_fft, hop_length,
            window=window, return_complex=True
        ).abs().clamp(min=epsilon)
        S_true = torch.stft(
            s_true.flatten(0, 1), n_fft, hop_length,
            window=window, return_complex=True
        ).abs().clamp(min=epsilon)

        L_sc = torch.sum(
            torch.sqrt(torch.sum((S_pred-S_true)**2, dim=(-2, -1)))
            / torch.sqrt(torch.sum(S_true**2, dim=(-2, -1)))
        )

        if spectral_magnitude_log:
            spec_diff = torch.log(S_pred) - torch.log(S_true)
        else:
            spec_diff = S_pred - S_true
        L_mag = torch.sum(
            torch.norm(torch.abs(spec_diff).clamp(min=epsilon),
                       p=spectral_magnitude_norm,
                       dim=(-2, -1))
        ) / (S_pred.shape[-2] * S_pred.shape[-1])

        loss += spectral_convergence_weight * L_sc \
            + spectral_magnitude_weight * L_mag

    return loss / (len(fft_spec) * s_pred.numel() / s_pred.shape[-1])

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

    return sdr.sum() / (s_pred.numel() / s_pred.shape[-1])
