#!/usr/env python
import os
import sys
import pathlib
import json
import logging
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from echidna import datasets as ds
from echidna import models as md
from echidna.models.waveunet import ChimeraWaveUNet
from echidna.models.convtasnet import ChimeraConvTasNet
from echidna.models.demucs import ChimeraDemucs
from echidna import metrics as mt

import torchaudio

losses_dc = [
    'deep-clustering',
    'whitened-kmeans',
    'reconstruction-wave-l1',
    'reconstruction-spec-l1',
]
losses_wa = [
    'wave-approximation-l1',
    'wave-approximation-l2',
    'multiscale-spectrum',
    'multiscale-spectrum-l1',
    'multiscale-spectrum-l2',
    'source-to-distortion-ratio',
    'scale-invariant-sdr',
    'scale-dependent-sdr',
]
losses = losses_dc + losses_wa

model_types = [
    'waveunet-based',
    'convtasnet-based',
    'demucs-based',
]

def parse_args():
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument('--model-type', choices=model_types, default=model_types[0])
    parser.add_argument('--model-params', type=str)

    # datasets
    parser.add_argument('--train-mix', required=True, nargs='+',
                        help='frozensample/mix-name')
    parser.add_argument('--train-samples', type=int, default=None)
    parser.add_argument('--validation-mix', nargs='*',
                        help='frozensample/mix-name')
    parser.add_argument('--validation-samples', type=int, default=None)
    parser.add_argument('--waveform-length', type=int, default=None)
    parser.add_argument('--loader-workers', type=int, default=0)
    parser.add_argument('--n-fft', type=int, default=2048)
    parser.add_argument('--hop-length', type=int, default=512)

    # training parameters
    parser.add_argument('--loss', nargs='+', choices=losses, required=True)
    parser.add_argument('--loss-weight', type=float, nargs='*')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--compute-batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad-norm', type=float, default=None)

    # checkpoint io
    parser.add_argument('--input-checkpoint', type=str, default=None)
    parser.add_argument('--output-checkpoint', type=str, required=True)
    parser.add_argument('--save-each-epoch', action='store_true')

    # misc
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--log')
    parser.add_argument('--log-level', default='INFO')

    args = parser.parse_args()

    try:
        args.model_params = json.loads(args.model_params)
    except:
        raise ValueError('--model-param needs to be json-formatted string')

    if args.compute_batch_size is None:
        args.compute_batch_size = args.batch_size

    if not args.loss_weight:
        args.loss_weight = [1.] * len(args.loss)
    if len(args.loss) != len(args.loss_weight):
        parser.error('losses and lambda must be the same number of parameters')

    args.disable_embedding = not any(l in losses_dc for l in args.loss)

    args.device = 'cuda' if args.gpu else 'cpu'

    return args

def build_separator(model_type, model_params):
    if model_type == 'waveunet-based':
        return ChimeraWaveUNet(**model_params)
    elif model_type == 'convtasnet-based':
        return ChimeraConvTasNet(**model_params)
    elif model_type == 'demucs-based':
        return ChimeraDemucs(**model_params)
    raise NotImplementedError(f'{model_type} model is not implemented')

def calculate_label(s, f, t, n_fft, hop_length):
    def spectrum_features(s):
        return torch.stft(
            s,
            n_fft,
            hop_length,
            window=torch.hann_window(n_fft).to(s.device),
            return_complex=True,
        ).abs()

    C = s.shape[1]
    feature = torch.stack([
        spectrum_features(_s.squeeze(1))
        for _s in s.split(1, dim=1)
    ], dim=-1) # BxFxTxC
    feature = torch.nn.functional.interpolate(feature.transpose(1, 3), (t, f))\
                                 .transpose(3, 1)
    label = torch.eye(s.shape[1], device=s.device)[feature.argmax(dim=-1)]
    return label

def compute_loss(pred, true, loss_fn_list, loss_weight_list, is_pit=True):
    # validation
    if type(pred.get('source', None)) != torch.Tensor:
        raise ValueError(
            'source requires in pred to calculate the loss function')
    if type(true.get('source', None)) != torch.Tensor:
        raise ValueError(
            'source requires in true to calculate the loss function')
    if any(l in losses_dc for l in loss_fn_list):
        if type(pred.get('embd', None)) != torch.Tensor:
            raise ValueError(
                'embd requires in pred to calculate the loss function')
        if type(true.get('label', None)) != torch.Tensor:
            true['label'] \
                = calculate_label(true['source'],
                                  *pred['embd'].shape[-3:-1],
                                  n_fft=2048,
                                  hop_length=512)

    loss_dict = dict()

    # calculate no-pit loss
    for fn in filter(lambda fn: fn in losses_dc, loss_fn_list):
        # deep clustering loss: the loss itself is permutation-invariant
        loss_dict[fn] = (
            mt.deep_clustering_loss(
                pred['embd'],
                true['label'],
                mode='deep-clustering',
            ) if fn == 'deep-clustering' else

            mt.deep_clustering_loss(
                pred['embd'],
                true['label'],
                mode='whitened-kmeans',
            ) if fn == 'whitened-kmeans' else

            mt.approximation_loss(
                pred['source'].sum(dim=1),
                true['source'].sum(dim=1),
                norm=1,
            ) if fn == 'reconstruction-wave-l1' else

            mt.multiscale_spectrogram_loss(
                pred['source'].sum(dim=1),
                true['source'].sum(dim=1),
                spectral_convergence_weight=0.,
                spectral_magnitude_weight = 1.,
                spectral_magnitude_norm=1,
            ) if fn == 'reconstruction-spec-l1' else

            0
        )

    # calculate pit loss
    wa_loss_fn_core = lambda fn, x, y: (
        # approximation loss
        mt.approximation_loss(x, y, norm=1)
        if fn == 'wave-approximation-l1' else

        mt.approximation_loss(x, y, norm=2)
        if fn == 'wave-approximation-l2' else

        # multiscale spectrum loss
        mt.multiscale_spectrogram_loss(x, y)
        if fn == 'multiscale-spectrum' else

        mt.multiscale_spectrogram_loss(
            x, y,
            spectral_convergence_weight=0.,
            spectral_magnitude_weight = 1.,
            spectral_magnitude_norm=1,
        ) if fn == 'multiscale-spectrum-l1' else

        mt.multiscale_spectrogram_loss(
            x, y,
            spectral_convergence_weight=0.,
            spectral_magnitude_weight = 1.,
            spectral_magnitude_norm=2,
        ) if fn == 'multiscale-spectrum-l2' else

        # loss for source to distortion ratio
        -mt.source_to_distortion_ratio(x, y)
        if fn == 'source-to-distortion-ratio' else

        -mt.source_to_distortion_ratio(
            x, y,
            scale_invariant=True,
        ) if fn == 'scale-invariant-sdr' else

        -mt.source_to_distortion_ratio(
            x, y,
            scale_dependent=True,
        ) if fn == 'scale-dependent-sdr' else

        0
    )

    # HACK: list of (weighted loss value, loss func name), (loss)
    def wa_loss_fn(x, y):
        loss_kv_list = [
            (fn, wa_loss_fn_core(fn, x, y))
            for fn in loss_fn_list if fn in losses_wa
        ]
        loss_kv = dict((k, v.detach()) for k, v in loss_kv_list)
        weighted_loss = sum([
            w * loss_kv[fn]
            for fn, w in zip(loss_fn_list, loss_weight_list)
            if fn in losses_wa
        ])
        return [('loss', weighted_loss), *loss_kv_list]

    if is_pit:
        pit_wrap = lambda fn: mt.permutation_invariant(
            fn,
            aggregate_perm_fn=min, # best permutation is minimum
            aggregate_loss_fn=lambda x: [
                (y[0][0], sum(z[1] for z in y) / len(y)) # find mean for each loss
                for y in list(zip(*x)) # -> ((L1,V11),(L1,V12)), ((L2,V21),(L2,V22))
            ]
        )
    else:
        pit_wrap = lambda fn: fn

    wa_loss_kv = pit_wrap(wa_loss_fn)(pred['source'], true['source'])
    if len(wa_loss_kv) > 1:
        loss_dict.update(dict(wa_loss_kv[1:]))

    return loss_dict

def process_batch(separator,
                  optimizer,
                  batch,
                  split_size,
                  loss_fn_list,
                  loss_weight,
                  n_fft,
                  hop_length,
                  max_grad_norm=1.0e6):

    sum_loss = dict((fn, 0) for fn in loss_fn_list)
    sum_loss['loss'] = 0

    if separator.training:
        optimizer.zero_grad()

    device = next(separator.parameters()).device

    for sample_i in range(0, batch.shape[0], split_size):
        # obtain batch and infer then get loss
        sample_i_end = min(sample_i + split_size, batch.shape[0])
        s = batch[sample_i:sample_i_end].to(device)
        x = s.sum(dim=1)

        # infer and transform
        model_ret = separator(x)
        if type(model_ret) == torch.Tensor:
            model_ret = (model_ret,)

        pred = dict(zip(('source', 'embd'), model_ret))
        true = {'source': s}
        # align true and pred pair
        if true['source'].shape[-1] > pred['source'].shape[-1]:
            begin_frame = (
                true['source'].shape[-1] - pred['source'].shape[-1]) // 2
            end_frame = begin_frame + pred['source'].shape[-1]
            true['source'] = true['source'][..., begin_frame:end_frame]
        elif true['source'].shape[-1] < pred['source'].shape[-1]:
            begin_frame = (
                pred['source'].shape[-1] - true['source'].shape[-1]) // 2
            end_frame = begin_frame + true['source'].shape[-1]
            pred['source'] = pred['source'][..., begin_frame:end_frame]
        if 'embd' in pred:
            true['label'] = calculate_label(
                s,
                separator.forward_embd_feature(),
                separator.forward_embd_length(s.shape[-1]),
                n_fft,
                hop_length,
            )

        logging.debug(
            'waveform stat: '
            + f'mean: {list(map(lambda x: round(x, 3), s.to("cpu").mean(dim=-1).flatten().tolist()))}, '
            + f'std: {list(map(lambda x: round(x, 3), s.to("cpu").std(dim=-1).flatten().tolist()))}, '
            + f'max: {list(map(lambda x: round(x, 3), s.to("cpu").abs().max(dim=-1)[0].flatten().tolist()))}'
        )
        logging.debug(
            'synth stat: '
            + f'mean: {list(map(lambda x: round(x, 3), model_ret[0].to("cpu").mean(dim=-1).flatten().tolist()))}, '
            + f'std: {list(map(lambda x: round(x, 3), model_ret[0].to("cpu").std(dim=-1).flatten().tolist()))}, '
            + f'max: {list(map(lambda x: round(x, 3), model_ret[0].to("cpu").abs().max(dim=-1)[0].flatten().tolist()))}'
        )
        if 'embd' in pred:
            logging.debug('embd stat: '
                          + f'{pred["embd"].shape}'
                          + f'{true["label"].shape}')

        # compute loss
        loss_dict = compute_loss(pred, true, loss_fn_list, loss_weight, is_pit=False)
        logging.debug(
            'loss stat:'
            + ' '.join(f'{ln}={ls.item():3g}' for ln, ls in loss_dict.items()),
        )

        for k, v in loss_dict.items():
            loss_dict[k] = v

        loss = sum(
            w * loss_dict[ln]
            for ln, w in zip(loss_fn_list, loss_weight)
        )

        for fn in loss_fn_list:
            sum_loss[fn] += loss_dict[fn].item() * s.shape[0]
        sum_loss['loss'] += loss.item() * s.shape[0]

        if separator.training:
            loss = loss * s.shape[0] / batch.shape[0]
            # backward
            loss.backward()
    # end for (for samples)

    if separator.training:
        # calculate gradnorm and clip
        total_norm = 0
        for name, p in separator.named_parameters():
            if p.grad is None or not p.requires_grad:
                #logging.debug(f'{name} does not have gradients')
                continue
            #norm = torch.sum(p.grad.detach().to('cpu').data ** 2)
            #logging.debug(f'{name} have gradients {norm}')
            total_norm += torch.sum(p.grad.detach().to('cpu').data ** 2)
        total_norm = torch.sqrt(total_norm).item()
        logging.debug(f'grad norm: {total_norm}')

        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(separator.parameters(), max_grad_norm)

        optimizer.step()

    return sum_loss

def save_checkpoint(output,
                    model_type,
                    model_params,
                    separator,
                    optimizer,
                    scheduler,
                    epoch,
                    **kwargs):
    device = next(separator.parameters()).device
    separator.to('cpu')
    for state in optimizer.state.values():
        for k, v in state.items():
            if type(v) == torch.Tensor:
                state[k] = v.to('cpu')

    # build object
    save_dict = {
        'model-type': model_type,
        'model-params': model_params,
        'model': separator.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'additional-info': kwargs,
    }

    # build path
    path_params = dict()
    if kwargs:
        path_params.update(kwargs)
    path_params['model-type'] = model_type
    path_params.update(model_params)
    path_params['epoch'] = epoch

    logging.info(f'saving model to {output.format(**path_params)}')
    torch.save(
        save_dict,
        pathlib.Path(output.format(**path_params)).expanduser()
    )

    separator.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if type(v) == torch.Tensor:
                state[k] = v.to(device)

def main():
    args = parse_args()
    if args.log is not None:
        logging.basicConfig(filename=args.log, level=args.log_level)
    else:
        logging.basicConfig(stream=sys.stdout, level=args.log_level)

    # build dataset
    train_datasets = [
        ds.FrozenMix(pathlib.Path(sampler_dir).expanduser(), mix_name)
        for sampler_dir, mix_name in [d.split(':') for d in args.train_mix]
    ]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    train_sampler = None
    if args.train_samples is not None \
       and args.train_samples < len(train_dataset):
        train_sampler = torch.utils.data.RandomSampler\
            (train_dataset, replacement=True, num_samples=args.train_samples)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.loader_workers,
        worker_init_fn=lambda: random.seed(),
        collate_fn=ds.collate_mix,
    )

    if args.validation_mix:
        validation_datasets = [
            ds.FrozenMix(pathlib.Path(sampler_dir).expanduser(), mix_name)
            for sampler_dir, mix_name in [
                    d.split(':') for d in args.validation_mix]
        ]
        validation_dataset = torch.utils.data.ConcatDataset(validation_datasets)
        if args.validation_samples is not None\
           and args.validation_samples < len(validation_dataset):
            indices = torch.linspace(0,
                                     len(validation_dataset)-1,
                                     args.validation_samples,
                                     dtype=int)
            validation_dataset = torch.utils.data.Subset(validation_dataset,
                                                         indices)
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=args.compute_batch_size,
            shuffle=False,
            num_workers=args.loader_workers,
            collate_fn=ds.collate_mix,
        )

    # build (and load) a model
    if args.input_checkpoint:
        cp = torch.load(pathlib.Path(args.input_checkpoint).expanduser())
        separator = build_separator(cp['model-type'], cp['model-params'])
        optimizer = torch.optim.Adam(
            separator.parameter_list(args.lr), args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5)
        separator.load_state_dict(cp['model'])
        if 'optimizer' in cp:
            optimizer.load_state_dict(cp.get('optimizer'))
        if 'scheduler' in cp:
            scheduler.load_state_dict(cp.get('scheduler'))
        init_epoch = cp['epoch'] + 1

    else:
        separator = build_separator(args.model_type, args.model_params)
        optimizer = torch.optim.Adam(
            separator.parameter_list(args.lr), args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5)
        init_epoch = 1

    epoch = init_epoch

    separator.to(args.device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if type(v) == torch.Tensor:
                state[k] = v.to(args.device)

    # print model statistics
    for k, v in separator.named_parameters():
        logging.debug(f'model size: {v.numel()}, shape: {v.shape} ({k})')

    # epoch loop
    for epoch in range(init_epoch, init_epoch+args.epoch):
        sum_loss = dict((ln, 0) for ln in args.loss)
        sum_loss['loss'] = 0
        total_batch = 0
        last_output_len = 0

        # training loop
        separator.train()
        for step, (data, metadata) in enumerate(train_loader, 1):
            batch = data['waveform']
            # obtain batch
            batch = batch.to(args.device)

            loss_dict = process_batch(separator,
                                      optimizer,
                                      batch,
                                      args.compute_batch_size,
                                      args.loss,
                                      args.loss_weight,
                                      args.n_fft,
                                      args.hop_length,
                                      args.grad_norm)

            total_batch += batch.shape[0]
            for ln, ls in loss_dict.items():
                sum_loss[ln] += ls

            # print learning statistics
            logging.info(
                f'epoch {epoch} step {step}: '
                + ' '.join(
                    f'{ln}={ls / total_batch:0.6g}'
                    for ln, ls in sum_loss.items()
                )
            )

        loss = dict(
            (ln, lv / total_batch)
            for ln, lv in sum_loss.items()
        )

        if args.validation_mix:
            # validation
            separator.eval()
            sum_val_loss = dict((ln, 0) for ln in args.loss)
            sum_val_loss['loss'] = 0
            total_val_batch = 0

            for (data, metadata) in validation_loader:
                batch = data['waveform']
                batch = batch.to(args.device)
                with torch.no_grad():
                    loss_dict = process_batch(separator,
                                              optimizer,
                                              batch,
                                              args.compute_batch_size,
                                              args.loss,
                                              args.loss_weight,
                                              args.n_fft,
                                              args.hop_length)

                total_val_batch += batch.shape[0]
                for ln, ls in loss_dict.items():
                    sum_val_loss[ln] += ls

            loss.update(dict(
                (f'val-{ln}', lv / total_val_batch)
                for ln, lv in sum_val_loss.items()
            ))

            # print learning statistics
            logging.info(
                f'epoch {epoch}:'
                + ', '.join(
                    f'{ln}={ls:0.6g}'
                    for ln, ls in loss.items()
                    if ln.startswith('val-')
                )
            )

        scheduler.step(loss['val-loss']
                       if 'val-loss' in loss else loss['loss'])

        # save checkpoint
        if args.save_each_epoch:
            save_checkpoint(args.output_checkpoint,
                            args.model_type,
                            args.model_params,
                            separator,
                            optimizer,
                            scheduler,
                            epoch,
                            **(loss if 'loss' in locals() else {}))

    if not args.save_each_epoch:
        save_checkpoint(args.output_checkpoint,
                        args.model_type,
                        args.model_params,
                        separator,
                        optimizer,
                        scheduler,
                        epoch,
                        **(loss if 'loss' in locals() else {}))

if __name__ == '__main__':
    main()

