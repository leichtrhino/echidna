
from datetime import datetime
import typing as tp
import json
import torch

from ..models.utils import match_length
from ..metrics.loss import get_loss_name
from ..metrics.composite import CompositeLoss

from . import trainings
from . import validations

class StepJournal(object):
    def __init__(self,
                 process_at : datetime,
                 step,
                 sample_losses : list,
                 batch_loss : float,
                 total_grad : float=None,
                 sample_indices : list=None):
        self.process_at = process_at
        self.step = step
        self.sample_losses = sample_losses
        self.batch_loss = batch_loss
        self.total_grad = total_grad
        self.sample_indices = sample_indices

    def to_dict(self):
        return {
            'process_at': self.process_at.isoformat(),
            'step': self.step,
            'sample_losses': self.sample_losses,
            'sample_indices': self.sample_indices,
            'batch_loss': self.batch_loss,
            'total_grad': self.total_grad
        }

    @classmethod
    def from_dict(cls, d : dict):
        return cls(
            process_at=datetime.fromisoformat(d['process_at']),
            step=d['step'],
            sample_losses=d['sample_losses'],
            sample_indices=d.get('sample_indices'),
            batch_loss=d['batch_loss'],
            total_grad=d.get('total_grad'),
        )


def process_batch(spec,
                  epoch : int,
                  step : int,
                  data : dict,
                  metadata : dict,
                  logger=None):

    assert type(spec) == trainings.TrainingSpec \
        or type(spec) == validations.ValidationSpec

    if type(spec) == trainings.TrainingSpec:
        checkpoint = spec.checkpoint
        model = checkpoint.get_torch_model()
        optimizer = checkpoint.get_torch_optimizer()
        model_class = checkpoint.get_model_class()
        model_epoch = checkpoint.get_epoch()
        mode = 'training' if model.training else 'validation'
    elif type(spec) == validations.ValidationSpec:
        model = spec.model.get_torch_model()
        model_class = spec.model.get_class()
        model_epoch = spec.model.get_epoch()
        mode = 'validation'

    batch_size = data['waves'].shape[0]
    compute_batch_size = spec.compute_batch_size
    loss_function = spec.loss_function
    device = next(model.parameters()).device

    # prepare variables
    if logger:
        event_dict = {
            'type': 'start_step',
            'mode': mode,
            'timestamp': datetime.now().isoformat(),
            'model_class': model_class,
            'model_epoch': model_epoch,
            'step': step,
        }
        if type(spec) == trainings.TrainingSpec:
            event_dict['training_epoch'] = epoch
        logger.info(json.dumps(event_dict))

    batch_loss = 0
    sample_losses = []

    if type(spec) == trainings.TrainingSpec and model.training:
        optimizer.zero_grad()

    for sample_i in range(0, batch_size, compute_batch_size):
        # obtain batch and infer then get loss
        sample_i_end = min(sample_i + compute_batch_size, batch_size)
        source = dict(
            (k, v[sample_i:sample_i_end].to(device))
            for k, v in data.items() if v is not None
        )
        x = source['waves'].sum(dim=1)
        if logger:
            event_dict = {
                'type': 'get_samples',
                'mode': mode,
                'timestamp': datetime.now().isoformat(),
                'model_class': model_class,
                'model_epoch': model_epoch,
                'step': step,
                'indices': metadata['index'][sample_i:sample_i_end],
            }
            if type(spec) == trainings.TrainingSpec:
                event_dict['training_epoch'] = epoch
            logger.info(json.dumps(event_dict))

        # infer
        pred = model(x)
        if logger:
            pred_swaps = []
            for si in range(sample_i_end - sample_i):
                pred_swaps.append(dict((k, v[si]) for k, v in pred.items()))
            event_dict = {
                'type': 'compute_inference',
                'mode': mode,
                'timestamp': datetime.now().isoformat(),
                'model_class': model_class,
                'model_epoch': model_epoch,
                'step': step,
                'stats': [
                    dict(
                        (k, {
                            'mean': s.mean(dim=-1).tolist(),
                            'std': s.std(dim=-1).tolist(),
                            'absmax': s.abs().max(dim=-1)[0].tolist(),
                        }) if k == 'waves' else
                        (k, {
                            'mean': s.mean().tolist(),
                            'std': s.std().tolist(),
                        }) if k == 'embd' else
                        None
                        for k, s in d.items()
                    )
                    for d in pred_swaps
                ]
            }
            if type(spec) == trainings.TrainingSpec:
                event_dict['training_epoch'] = epoch
            logger.info(json.dumps(event_dict))

        # align waveform and pred
        if 'waves' in pred:
            wave_length = min(source['waves'].shape[-1],
                              pred['waves'].shape[-1])
            source['waves'] = match_length(source['waves'], wave_length)
            pred['waves'] = match_length(pred['waves'], wave_length)
        # add embedding
        if 'embd' in pred:
            feature = torch.stack([
                torch.stft(
                    s.squeeze(1),
                    spec.n_fft,
                    spec.hop_length,
                    window=torch.hann_window(spec.n_fft).to(s.device),
                    return_complex=True,
                ).abs()
                for s in source['waves'].split(1, dim=1)
            ], dim=-1) # BxFxTxC
            feature = torch.nn.functional\
                              .interpolate(
                                  feature.transpose(1, 3),
                                  pred['embd'].shape[1:3]
                              )\
                              .transpose(3, 1)
            source['embd'] = feature

        if logger:
            source_swaps = []
            for si in range(sample_i_end - sample_i):
                source_swaps.append(
                    dict((k, v[si]) for k, v in source.items()))
            event_dict = {
                'type': 'align_samples',
                'mode': mode,
                'timestamp': datetime.now().isoformat(),
                'model_class': model_class,
                'model_epoch': model_epoch,
                'step': step,
                'stats': [
                    dict(
                        (k, {
                            'mean': s.mean(dim=-1).tolist(),
                            'std': s.std(dim=-1).tolist(),
                            'absmax': s.abs().max(dim=-1)[0].tolist(),
                        }) if k == 'waves' else
                        (k, {
                            'mean': s.mean().tolist(),
                            'std': s.std().tolist(),
                        }) if k == 'embd' else
                        None
                        for k, s in d.items()
                    )
                    for d in source_swaps
                ]
            }
            if type(spec) == trainings.TrainingSpec:
                event_dict['training_epoch'] = epoch
            logger.info(json.dumps(event_dict))

        # compute loss
        if type(loss_function) == CompositeLoss:
            loss_value = loss_function.forward_no_reduction(pred, source)
            loss_values = {
                'batch': loss_value['batch'].mean(),
                'sample': loss_value['sample'],
            }
        else:
            domain = loss_function.domains[0]
            loss_value = loss_function.forward_no_reduction(
                pred[domain], source[domain])
            loss_values = {
                'batch': loss_value.mean(),
                'sample': [
                    {get_loss_name(type(loss_function)): l}
                    for l in loss_value.tolist()
                ]
            }

        batch_loss += loss_values['batch'].item() \
            * (sample_i_end - sample_i) / batch_size
        sample_losses.extend(loss_values['sample'])
        if logger:
            event_dict = {
                'type': 'compute_loss',
                'mode': mode,
                'timestamp': datetime.now().isoformat(),
                'model_class': model_class,
                'model_epoch': model_epoch,
                'step': step,
                'losses': loss_values['sample'],
            }
            if type(spec) == trainings.TrainingSpec:
                event_dict['training_epoch'] = epoch
            logger.info(json.dumps(event_dict))

        # backward
        if type(spec) == trainings.TrainingSpec and model.training:
            # NOTE: to avoid the instability of learning,
            #       scale the batch size from specification
            #       (the most case is batch.shape[0] <= spec.batch_size)
            sum_loss = loss_values['batch'] \
                * (sample_i_end - sample_i) / spec.batch_size
            sum_loss.backward()

    # end for (for samples)

    # calculate gradnorm and update
    grad_norm = None
    if type(spec) == trainings.TrainingSpec and model.training:
        grad_norm = 0
        layer_grad_square_sum = dict()
        for name, p in model.named_parameters():
            if p.grad is None or not p.requires_grad:
                continue
            grad_square_sum = torch.sum(p.grad.detach().to('cpu').data ** 2)
            grad_norm += grad_square_sum
            layer_grad_square_sum[name] = grad_square_sum.item()
        grad_norm = torch.sqrt(grad_norm).item()

        if spec.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           spec.max_grad_norm)
        optimizer.step()
        if logger:
            event_dict = {
                'type': 'update_model',
                'mode': mode,
                'timestamp': datetime.now().isoformat(),
                'model_class': model_class,
                'model_epoch': model_epoch,
                'step': step,
                'total_grad_norm': grad_norm,
            }
            if type(spec) == trainings.TrainingSpec:
                event_dict['training_epoch'] = epoch
            logger.info(json.dumps(event_dict))

            event_dict = {
                'type': 'update_model_detail',
                'mode': mode,
                'timestamp': datetime.now().isoformat(),
                'model_class': model_class,
                'model_epoch': model_epoch,
                'step': step,
                'grad_square_sums': layer_grad_square_sum,
            }
            if type(spec) == trainings.TrainingSpec:
                event_dict['training_epoch'] = epoch
            logger.debug(json.dumps(event_dict))

    # build step journal
    step_journal = StepJournal(datetime.now(),
                               step,
                               sample_losses=sample_losses,
                               batch_loss=batch_loss,
                               total_grad=grad_norm,
                               sample_indices=metadata['index'])
    if logger:
        event_dict = {
            'type': 'end_step',
            'mode': mode,
            'timestamp': datetime.now().isoformat(),
            'model_class': model_class,
            'model_epoch': model_epoch,
            'step': step,
        }
        if type(spec) == trainings.TrainingSpec:
            event_dict['training_epoch'] = epoch
        logger.info(json.dumps(event_dict))


    return step_journal

