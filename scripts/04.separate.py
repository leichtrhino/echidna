#!/usr/env python
import os
import sys
import logging
import itertools
import math
import argparse
import resampy
import torch
import torchaudio
import sklearn.cluster

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from chimerau.models.waveunet import ChimeraWaveUNet
from chimerau.models.convtasnet import ChimeraConvTasNet
from chimerau.models.demucs import ChimeraDemucs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', nargs='+', required=True)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--algorithm', choices=['embedding', 'waveform'], default='waveform')
    parser.add_argument('--sr', type=int)
    parser.add_argument('--n-fft', type=int, default=1024, required=True)
    parser.add_argument('--hop-length', type=int, default=256, required=True)
    parser.add_argument('--duration', type=float, default=5.)
    parser.add_argument('--overlap', type=float, default=0.5)

    args = parser.parse_args()

    if args.overlap < 0:
        raise ValueError('--overlap must be >= 0')
    elif args.overlap >= 1:
        raise ValueError('--overlap must be < 1')

    if args.gpu:
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    return args

def build_separator(model_type, model_params):
    if model_type == 'waveunet-based':
        return ChimeraWaveUNet(**model_params)
    elif model_type == 'demucs-based':
        return ChimeraDemucs(**model_params)
    elif model_type == 'convtasnet-based':
        return ChimeraConvTasNet(**model_params)
    raise NotImplementedError(f'{model_type} model is not implemented')

def split_into_windows(x, win_length, hop_length):
    sample_num = max(0, math.ceil((x.shape[-1] - win_length) / hop_length)) + 1
    sample = []
    for si in range(sample_num):
        left = si * hop_length
        right = min(left + win_length, x.shape[-1])
        x_part = x[..., left:right]
        if x_part.shape[-1] < win_length:
            x_part = torch.cat((
                x_part,
                torch.zeros(
                    *x_part.shape[:-1],
                    win_length - x_part.shape[-1]
                )
            ), dim=-1)
        sample.append(x_part)
    sample = torch.stack(sample, dim=-2)
    return sample

def merge_windows(x, hop_length, agg_fn='mean'):
    '''
    x: torch.Tensor shape with (window_num, channel_num, window_len)
    or (window_num, channel_num, feature_num, window_len)
    hop_length: int
    agg_fn: str
        'mean' or 'mode' or 'median'

    returns: torch.Tensor shape with (channel_num, *, output_length)
    '''
    if agg_fn not in ['mean', 'mode', 'median']:
        raise ValueError(f'agg_fn needs to be "mean" or "mode"')

    # initialize base and index tensor its shape is
    # (*x.shape[1:-1], (window_num-1)*hop_length+window_len, n_overlap)
    # where n_overlap=ceil(window_len / hop_length)
    # and values are filled with NaN
    # initialize apperence list with length=base.shape[-2]
    channels = x.shape[1]
    win_length = x.shape[-1]
    out_length = (x.shape[0] - 1) * hop_length + win_length
    overlaps = math.ceil(win_length / hop_length) + 1
    base = torch.full((*x.shape[1:-1], out_length, overlaps), torch.nan, dtype=x.dtype)
    #indices = [[] for _ in range(out_length)]

    # replace the tensor value with x
    for wi, w in enumerate(x):
        overlap_i = wi % overlaps
        out_begin = min(wi * hop_length, out_length)
        out_end = wi * hop_length + win_length

        # find best permutation along axis=1 (channel)
        base_part = base[..., out_begin:out_end, :]
        p_best = list(range(channels))
        p_dist_best = torch.inf
        """
        for p in itertools.permutations(list(range(channels))):
            if agg_fn == 'mean' or agg_fn == 'median':
                # l1 distance for agg_fn='mean' or 'median'
                p_dist = torch.nansum(torch.abs(
                    w[p, ...][..., 0:out_end-out_begin].unsqueeze(-1)
                    - base_part
                ))
            elif agg_fn == 'mode':
                # hamming distance for agg_fn='mode'
                p_dist = torch.nansum(
                    w[p, ...][..., 0:out_end-out_begin].unsqueeze(-1)
                    != base_part,
                ).to(dtype=torch.float)
            print(f'permutation {p}: {p_dist}')
            if p_dist < p_dist_best:
                p_best = p
                p_dist_best = p_dist
        """

        print('best permutation', p_best)
        base[..., out_begin:out_end, overlap_i] \
            = w[p_best, ...][..., 0:out_end-out_begin]
        #for fi in range(out_begin, out_end):
        #    indices[fi].append(overlap_i)

    # aggregate
    print(base.shape)
    if agg_fn == 'mean':
        out_tensor = torch.nansum(base, dim=-1) \
            / (overlaps - torch.sum(torch.isnan(base), dim=-1))
    elif agg_fn == 'median':
        out_tensor, _ = torch.nanmedian(base, dim=-1)
    elif agg_fn == 'mode':
        out_tensor, _ = torch.mode(base, dim=-1)

    """
    out_tensor = torch.empty((*x.shape[1:-1], out_length), dtype=x.dtype)
    for fi, overlap_idx in enumerate(indices):
        frame = base[..., fi, overlap_idx]
        if agg_fn == 'mean':
            agg_frame = torch.mean(frame, axis=-1)
        elif agg_fn == 'median':
            agg_frame, _ = torch.median(frame, axis=-1)
        elif agg_fn == 'mode':
            agg_frame, _ = torch.mode(frame, axis=-1)
        if torch.isnan(agg_frame).sum().item() > 0:
            logging.warning(f'NaN found in frame {fi}, replace to 0')
            agg_frame = torch.nan_to_num(agg_frame)
        out_tensor[..., fi] = agg_frame
    """

    return out_tensor

def mask_from_embedding(embedding, n_channels, out_F, out_T):
    '''
    embedding: torch.Tensor with shape (N, F, T, D)
    returns: torch.Tensor with shape (N, C, F, T)
    '''
    F, T, D = embedding.shape[-3:]
    embedding = torch.nn.functional.interpolate(
        embedding.transpose(-3, -1), # -> (N, D, T, F)
        (out_T, out_F)
    ).transpose(-3, -1) # -> (N, F, T, D)
    embedding = embedding / torch.norm(embedding, dim=-1, keepdim=True)

    masks = []
    for e in embedding:
        labels = torch.from_numpy(
            sklearn.cluster.KMeans(n_clusters=n_channels)
            .fit(e.flatten(0, 1).numpy())
            .labels_
        ).to(dtype=torch.long)
        mask = torch.eye(n_channels)[labels]\
            .transpose(0, 1)\
            .unflatten(-1, (out_F, out_T))\
            .to(dtype=torch.float)
        masks.append(mask)
    return torch.stack(masks, dim=0)

'''
    labels = sklearn.cluster.KMeans(
        n_clusters=n_channels, n_jobs=n_jobs
    ).fit(
        embedding.reshape(embedding.size // embedding_dim, embedding_dim)
    ).labels_
    mask = np.eye(n_channels)[labels]\
        .reshape(list(embedding.shape[:-1])+[n_channels])\
        .transpose((0, 3, 2, 1))
    return mask
'''

def process_channel(x):
    pass


def main():
    args = parse_args()

    # build (and load) a model
    cp = torch.load(args.checkpoint)
    separator = build_separator(cp['model-type'], cp['model-params'])
    separator.load_state_dict(cp['model'])
    if args.algorithm == 'waveform':
        separator = separator.get_core_model()
    separator.eval()
    separator.to(args.device)

    # load audio file
    x, orig_sr = torchaudio.load(args.input)
    if args.sr is not None:
        x = torch.Tensor(resampy.resample(
            x.numpy(),
            orig_sr,
            args.sr,
            axis=-1
        ))
        sr = args.sr
    else:
        sr = orig_sr

    # 1. find output_waveform per sample
    # 2. find output_hop_length of output_waveform
    # 3. find input_waveform per sample (using reverse length)
    # 4. find left-extend and right-extend
    # 5. find length of padded input waveform
    # example of out_length=10, overlap=50%
    # |-- l_in --|
    #      |-- l_in --|
    #           |-- l_in --|
    #                |-- l_in --|
    #  |- l_out-|
    #       |- l_out-|
    #            |- l_out-|
    #                 |- l_out-|
    # total length of input: total length of output
    #                        + rev_len(out_sample_len) - out_sample_len
    orig_length = x.shape[-1]
    out_sample_len = int(sr * args.duration)
    sample_hop_len = int(sr * args.duration * (1 - args.overlap))

    sample_num = math.ceil((orig_length - out_sample_len) / sample_hop_len) + 1
    in_sample_len = separator.reverse_length(out_sample_len)
    expand_left = (in_sample_len - out_sample_len) // 2
    expand_right = (sample_num - 1) * sample_hop_len + out_sample_len \
        + in_sample_len - (expand_left + out_sample_len) - x.shape[-1]

    padded_x = torch.cat((
        torch.zeros(*x.shape[:-1], expand_left),
        x,
        torch.zeros(*x.shape[:-1], expand_right)
    ), dim=-1)
    sample = split_into_windows(padded_x, in_sample_len, sample_hop_len)

    # separate
    out_tensors = []
    for channel_i, channel in enumerate(sample):
        out_windows = None
        for batch_i in range(0, channel.shape[0], args.batch_size):
            batch_end_i = min(batch_i + args.batch_size, channel.shape[0])
            batch = channel[batch_i:batch_end_i]
            if args.algorithm == 'waveform':
                in_l = batch.shape[-1]
                out_l = separator.forward_length(batch.shape[-1])
                el = (in_l - out_l) // 2
                er = in_l - out_l - el

                # NOTE: the program requires input channels = 1
                # loop until the output channel reaches the disired channels
                bc_tensor = batch.unsqueeze(1)
                while bc_tensor.shape[1] < len(args.output):
                    # find s_hat and append, then, drop first channel
                    with torch.no_grad():
                        s_hat = separator(
                            bc_tensor[:, 0, :].to(args.device)).to('cpu')
                    assert type(s_hat) == torch.Tensor
                    # NOTE: alignments is zero-padded
                    # TODO: sort by entropy of separation
                    #       and magnitude of specgram
                    s_hat = torch.cat((
                        torch.zeros(*s_hat.shape[:-1], el),
                        s_hat,
                        torch.zeros(*s_hat.shape[:-1], er)
                    ), dim=-1)
                    if len(s_hat.shape) < 2:
                        s_hat = s_hat.unsqueeze(1)
                    bc_tensor = torch.cat((bc_tensor, s_hat), dim=1)\
                        [:, 1:, :]
                    # sort by sum of spectrogram
                    """
                    bc_tensor = torch.stack([
                        torch.stack(
                            sorted(
                                cs,
                                key=lambda c: torch.stft(
                                    c,
                                    n_fft=args.n_fft,
                                    hop_length=args.hop_length,
                                    return_complex=True
                                ).abs().sum(),
                                reverse=True
                            ),
                            dim=0
                        )
                        for cs in bc_tensor # loop along batch
                    ], dim=0)
                    """
                out_net = bc_tensor[:, :len(args.output), el:el+out_l]

            elif args.algorithm == 'embedding':
                out_l = separator.forward_length(batch.shape[-1])
                # XXX: actually, the second output of the net is embedding
                out_F, out_T = torch.stft(
                    torch.ones(out_l),
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    return_complex=True
                ).shape
                with torch.no_grad():
                    embd = separator(batch.to(args.device))[1].to('cpu')
                out_net = mask_from_embedding(embd, len(args.output), out_F, out_T)

            if type(out_windows) != torch.Tensor:
                out_windows = out_net
            else:
                out_windows = torch.cat((out_windows, out_net), dim=0)

        if args.algorithm == 'waveform':
            aligned_channel = merge_windows(
                out_windows,
                sample_hop_len,
                agg_fn='median'
            )
            print('aligned_channel:',
                  aligned_channel.mean(dim=-1),
                  aligned_channel.std(dim=-1))

        # TODO: un-window channel and out_windows separately
        # and then multiply
        if args.algorithm == 'embedding':
            in_l = channel.shape[-1]
            out_l = separator.forward_length(channel.shape[-1])
            el = (in_l - out_l) // 2
            er = el + out_l

            base = torch.stft(
                channel[..., el:er],
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                return_complex=True,
            )
            aligned_channel = merge_windows(
                out_windows * base.unsqueeze(1),
                int(out_windows.shape[-1] * (1 - args.overlap)),
                agg_fn='mean'
            )

            aligned_channel = torch.istft(
                aligned_channel,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
            )

        out_tensors.append(aligned_channel)

    s_hats = torch.stack(out_tensors)[..., :orig_length]
    # TODO: find best channel pertition

    # save
    for out_file, s_hat in zip(args.output, s_hats.transpose(0, 1)):
        torchaudio.save(out_file, s_hat, sample_rate=sr)

if __name__ == '__main__':
    main()

