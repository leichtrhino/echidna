
import os
import math
import itertools
import json
import logging
import torch
import torchaudio
from datetime import datetime

from ..models.models import Model
from ..data.transforms import Resample

class SeparationSpec(object):
    def __init__(self,
                 # model
                 model : Model,
                 # input/output
                 input : str,
                 output : list,
                 journal_pattern : str,
                 log_pattern : str,
                 log_level : str,
                 # separation quality parameters
                 sample_rate : int,
                 duration : float,
                 overlap : float,
                 permutation_invariant : bool,
                 # separation process parameters
                 batch_size : int,
                 device : str,
                 ):

        if model.get_torch_model().forward_wave_channel() != len(output):
            model_channel = model.get_torch_model().forward_wave_channel()
            file_channel = len(output)
            raise ValueError(
                f'the output files ({file_channel}) and '
                f'model channels ({model_channel}) are different')

        # model
        self.model = model
        # input/output
        self.input = input
        self.output = output
        self.journal_pattern = journal_pattern
        self.log_pattern = log_pattern
        self.log_level = log_level
        # separation quality parameters
        self.sample_rate = sample_rate
        self.duration = duration
        self.overlap = overlap
        self.permutation_invariant = permutation_invariant
        # separation process parameters
        self.batch_size = batch_size
        self.device = device

        # check journal_pattern and log_pattern
        self._validate_patterns()

    @classmethod
    def from_dict(cls, d : dict):
        return cls(
            # model
            model=Model.from_dict(d['model']),
            # input/output
            input=d['input'],
            output=d['output'],
            journal_pattern=d.get('journal_pattern', None),
            log_pattern=d.get('log_pattern', None),
            log_level=d.get('log_level', 'INFO'),
            # separation quality parameters
            sample_rate=d['sample_rate'],
            duration=d.get('duration', None),
            overlap=d.get('overlap', None),
            permutation_invariant=d.get('permutation_invariant', False),
            # computation parameters
            batch_size=d.get('batch_size', 1),
            device=d.get('device', 'cpu'),
        )

    def to_dict(self):
        return {
            # model
            'model': self.model.to_dict(),
            # input/output
            'input': self.input,
            'output': self.output,
            'journal_pattern': self.journal_pattern,
            'log_pattern': self.log_pattern,
            'log_level': self.log_level,
            # separation quality parameters
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'overlap': self.overlap,
            'permutation_invariant': self.permutation_invariant,
            # separation process parameters
            'batch_size': self.batch_size,
            'device': self.device,
        }

    def _validate_patterns(self):
        template_before_epoch = {
            # from specification
            'model': self.model.get_class(),
            'model_epoch': self.model.get_epoch(),
            # from epoch journal
            'process_start': datetime.now(),
        }
        template_after_epoch = {
            # from specification
            'model': self.model.get_class(),
            'model_epoch': self.model.get_epoch(),
            # from epoch journal
            'process_start': datetime.now(),
            'process_finish': datetime.now(),
        }

        if self.log_pattern:
            self.log_pattern.format(**template_before_epoch)
        if self.journal_pattern:
            self.journal_pattern.format(**template_after_epoch)

    def separate(self):
        _separate(self)

class SeparationJournal(object):
    def __init__(self,
                 process_start : datetime,
                 process_finish : datetime,
                 log_path : str,
                 spec : SeparationSpec,
                 ):

        self.process_start = process_start
        self.process_finish = process_finish
        self.log_path = log_path
        self.spec = spec

    def to_dict(self):
        return {
            'process_start': self.process_start.isoformat(),
            'process_finish': self.process_finish.isoformat(),
            'log_path': self.log_path,
            'spec': self.spec.to_dict(),
        }

    @classmethod
    def from_dict(cls, d : dict):
        return cls(
            process_start=datetime.fromisoformat(d['process_start']),
            process_finish=datetime.fromisoformat(d['process_finish']),
            log_path=d.get('log_path'),
            spec=SeparationSpec.from_dict(d['spec']),
        )

def _separate(spec : SeparationSpec):

    process_start = datetime.now()

    # get log path and initialize log handler
    logger = None
    if spec.log_pattern:
        pattern_dict ={
            # from specification
            'model': spec.model.get_class(),
            'model_epoch': spec.model.get_epoch(),
            # from separation journal
            'process_start': process_start,
        }

        logger = logging.getLogger(__name__)
        log_path = spec.log_pattern.format(**pattern_dict)
        logger.setLevel(spec.log_level)
        if not os.path.isdir(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(
            logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)

    if logger:
        logger.info(json.dumps({
            'type': 'start_separation',
            'timestamp': datetime.now().isoformat(),
            'model_class': spec.model.get_class(),
            'model_epoch': spec.model.get_epoch(),
        }))

    # get torch model from specification
    model = spec.model.get_torch_model()
    model.eval()
    model.to(spec.device)

    # load audio file
    x, orig_sr = torchaudio.load(spec.input)
    if spec.sample_rate is not None:
        x = Resample(orig_sr, spec.sample_rate)(x)
        out_sample_rate = spec.sample_rate
    else:
        out_sample_rate = orig_sr

    if logger:
        logger.info(json.dumps({
            'type': 'load',
            'timestamp': datetime.now().isoformat(),
            'model_class': spec.model.get_class(),
            'model_epoch': spec.model.get_epoch(),
            'input': spec.input,
            'waveform_length': x.shape,
        }))

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
    if spec.duration:
        out_sample_len = int(out_sample_rate * spec.duration)
        sample_hop_len = int(out_sample_rate * spec.duration
                             * (1 - spec.overlap))

        sample_num = math.ceil((orig_length - out_sample_len)
                               / sample_hop_len) + 1
        in_sample_len = model.reverse_wave_length(out_sample_len)
        expand_left = (in_sample_len - out_sample_len) // 2
        expand_right = (sample_num - 1) * sample_hop_len + out_sample_len \
            + in_sample_len - (expand_left + out_sample_len) - x.shape[-1]

        padded_x = torch.cat((
            x,
            torch.zeros(*x.shape[:-1], expand_left+expand_right)
        ), dim=-1)
        sample = split_into_windows(padded_x, in_sample_len, sample_hop_len)
    else:
        sample = x.unsqueeze(1)
        in_sample_len = model.reverse_wave_length(orig_length)
        if in_sample_len > orig_length:
            expand_left = (in_sample_len - orig_length) // 2
            expand_right = in_sample_len - (expand_left + orig_length)
            sample = torch.cat((
                sample,
                torch.zeros(*sample.shape[:-1], expand_left+expand_right),
            ), dim=-1)

    if logger:
        logger.info(json.dumps({
            'type': 'split',
            'timestamp': datetime.now().isoformat(),
            'model_class': spec.model.get_class(),
            'model_epoch': spec.model.get_epoch(),
            'num_channel': len(sample),
            'num_window': len(sample[0]),
        }))

    # separate
    out_tensors = []
    for in_channel_i, channel in enumerate(sample):
        out_windows = []
        for batch_i in range(0, channel.shape[0], spec.batch_size or 1):
            batch_end_i = min(batch_i + (spec.batch_size or 1),
                              channel.shape[0])
            batch = channel[batch_i:batch_end_i]

            # infer source waveform for an input channel and a window
            with torch.no_grad():
                s_hat = model(batch.to(spec.device))['waves'].to('cpu')
            out_windows.extend(s_hat)

            if logger:
                logger.debug(json.dumps({
                    'type': 'separate_progress',
                    'timestamp': datetime.now().isoformat(),
                    'model_class': spec.model.get_class(),
                    'model_epoch': spec.model.get_epoch(),
                    'proceeded_samples':
                    in_channel_i * len(sample[0]) + batch_end_i,
                    'sample_shape': tuple(s_hat.shape),
                }))

        if spec.duration:
            aligned_channel = merge_windows(
                torch.stack(out_windows, dim=0),
                sample_hop_len,
                agg_fn='median',
                permutation_invariant=spec.permutation_invariant,
            )
        else:
            aligned_channel = torch.cat(out_windows, dim=-1)
        out_tensors.append(aligned_channel)

        if logger:
            logger.debug(json.dumps({
                'type': 'separate_progress',
                'timestamp': datetime.now().isoformat(),
                'model_class': spec.model.get_class(),
                'model_epoch': spec.model.get_epoch(),
                'proceeded_channel': in_channel_i,
                'channel_shape': tuple(aligned_channel.shape),
            }))

    s_hats = torch.stack(out_tensors)
    left = (s_hats.shape[-1] - orig_length) // 2
    s_hats = s_hats[..., :orig_length]

    if logger:
        logger.info(json.dumps({
            'type': 'separate',
            'timestamp': datetime.now().isoformat(),
            'model_class': spec.model.get_class(),
            'model_epoch': spec.model.get_epoch(),
            'proceeded_samples': len(sample) * len(sample[0]),
            'waveform_shape': tuple(s_hats.shape),
        }))

    # save
    for out_file, s_hat in zip(spec.output, s_hats.transpose(0, 1)):
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
        torchaudio.save(out_file, s_hat, sample_rate=out_sample_rate)

    if logger:
        logger.info(json.dumps({
            'type': 'save',
            'timestamp': datetime.now().isoformat(),
            'model_class': spec.model.get_class(),
            'model_epoch': spec.model.get_epoch(),
            'output': spec.output,
        }))

    process_finish = datetime.now()

    # save journal
    if spec.journal_pattern:
        pattern_dict['process_finish'] = process_finish
        journal_path = spec.journal_pattern.format(**pattern_dict)
        journal = SeparationJournal(
            process_start=process_start,
            process_finish=process_finish,
            log_path=log_path,
            spec=spec,
        )
        if not os.path.isdir(os.path.dirname(journal_path)):
            os.makedirs(os.path.dirname(journal_path), exist_ok=True)
        with open(journal_path, 'w') as fp:
            json.dump(journal.to_dict(), fp)

        logger.info(json.dumps({
            'type': 'save_journal',
            'timestamp': datetime.now().isoformat(),
            'model_class': spec.model.get_class(),
            'model_epoch': spec.model.get_epoch(),
            'journal_path': journal_path,
        }))

    # finish epoch and close log handler
    if logger:
        logger.info(json.dumps({
            'type': 'end_separation',
            'timestamp': datetime.now().isoformat(),
            'model_class': spec.model.get_class(),
            'model_epoch': spec.model.get_epoch(),
        }))
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()

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

def merge_windows(x,
                  hop_length,
                  agg_fn='mean',
                  permutation_invariant=False
                  ):
    """
    Parameters
    ----------
    x: torch.Tensor shape with (window_num, channel_num, window_len)
    or (window_num, channel_num, feature_num, window_len)
    hop_length: int
    agg_fn: str
        'mean' or 'mode' or 'median'
    permutation_invariant: bool

    Returns
    -------
    torch.Tensor shape with (channel_num, *, output_length)
    """
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

    # replace the tensor value with x
    for wi, w in enumerate(x):
        overlap_i = wi % overlaps
        out_begin = min(wi * hop_length, out_length)
        out_end = wi * hop_length + win_length

        # find best permutation along axis=1 (channel)
        base_part = base[..., out_begin:out_end, :]
        perm = list(range(channels))
        if permutation_invariant:
            p_best = perm
            p_dist_best = torch.inf
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
                if p_dist < p_dist_best:
                    p_best = p
                    p_dist_best = p_dist
            perm = p_best

        base[..., out_begin:out_end, overlap_i] \
            = w[perm, ...][..., 0:out_end-out_begin]

    # aggregate
    if agg_fn == 'mean':
        out_tensor = torch.nansum(base, dim=-1) \
            / (overlaps - torch.sum(torch.isnan(base), dim=-1))
    elif agg_fn == 'median':
        out_tensor, _ = torch.nanmedian(base, dim=-1)
    elif agg_fn == 'mode':
        out_tensor, _ = torch.mode(base, dim=-1)

    return out_tensor

