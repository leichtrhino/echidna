
import typing as tp
from itertools import permutations
import torch

from .loss import Loss, register_loss_class, get_loss_name

class CompositeLoss(Loss):
    """
    NOTE: reduction attribute of each loss is ignored

    Attributes
    ----------
    components : list
        list of dictionary that contains func,
        param(optional), weight(optional)
    """

    def __init__(self,
                 components : list,
                 permutation='none',
                 reduction='mean'):
        super().__init__(reduction)
        if any(type(d) != dict or not isinstance(d.get('func'), Loss)
               for d in components):
            raise ValueError('all elements in components must have func')
        if permutation not in ('none', 'min', 'max'):
            raise ValueError(f'invalid permutation mode {permutation}')
        self.components = components
        self.permutation = permutation

    @property
    def domains(self):
        return set(dm for d in self.components for dm in d['func'].domains)

    def to_dict_args(self):
        return {
            'components': [
                {
                    'func': d['func'].to_dict(),
                    'weight': d.get('weight', 1.0),
                }
                for d in self.components
            ],
            'permutation': self.permutation,
            'reduction': self.reduction
        }

    @classmethod
    def from_dict_args(cls, d : dict):
        return cls(
            components=[
                {
                    'func': Loss.from_dict(e['func']),
                    'weight': e.get('weight', 1.0)
                }
                for e in d['components']
            ],
            permutation=d.get('permutation', 'none'),
            reduction=d.get('reduction', 'mean')
        )

    def _forward_no_reduction_with_perm(self,
                                        p_pred : dict,
                                        p_true : dict,
                                        cache : list):

        batch_size = next(iter(p_pred.values())).shape[0]
        losses = {
            'sample': [dict() for _ in range(batch_size)],
            'batch': None
        }

        for ci, c in enumerate(self.components):
            loss_fn = c['func']
            domains = c['func'].domains
            weight = c['weight']

            if 'waves' not in domains \
               and cache is not None and cache[ci] is not None:
                loss = cache[ci]
            else:
                if len(domains) == 1:
                    pred, true = p_pred[domains[0]], p_true[domains[0]]
                else:
                    pred, true = p_pred, p_true
                loss = loss_fn.forward_no_reduction(pred, true)
                if cache is not None:
                    cache[ci] = loss

            # append loss to sample and batch
            for li, lv in enumerate(loss.tolist()):
                losses['sample'][li][get_loss_name(type(loss_fn))] = lv
            if losses['batch'] is None:
                losses['batch'] = weight * loss
            else:
                losses['batch'] = losses['batch'] + weight * loss

        return losses

    def forward_no_reduction(self, y_pred : dict, y_true : dict):
        if type(y_pred) != dict or any(d not in y_pred for d in self.domains):
            raise ValueError(f'y_pred does not contain {self.domains}')
        if type(y_true) != dict or any(d not in y_true for d in self.domains):
            raise ValueError(f'y_true does not contain {self.domains}')

        domains = self.domains
        if self.permutation == 'none' or 'waves' not in domains:
            losses = self._forward_no_reduction_with_perm(y_pred,
                                                          y_true,
                                                          cache=None)
            return losses

        # for permutation-invariant loss
        n_batch, n_channel = y_pred['waves'].shape[:2]
        losses = {
            'sample': [],
            'batch': []
        }
        for bi in range(n_batch):
            b_pred = dict((k, v[bi].unsqueeze(0)) for k, v in y_pred.items())
            b_true = dict((k, v[bi].unsqueeze(0)) for k, v in y_true.items())
            cache = [None for _ in range(len(self.components))]

            best_losses = None
            for perm in permutations(range(n_channel)):

                p_pred = dict((k, v)
                              for k, v in b_pred.items() if k != 'waves')
                p_pred['waves'] = b_pred['waves'][:, list(perm), ...]
                p_true = dict((k, v) for k, v in b_true.items())

                perm_losses = self._forward_no_reduction_with_perm(p_pred,
                                                                   p_true,
                                                                   cache)
                B = 'batch'
                if best_losses is None \
                   or (
                       self.permutation == 'min'
                       and perm_losses[B].sum() < best_losses[B].sum()
                       or self.permutation == 'max'
                       and perm_losses[B].sum() > best_losses[B].sum()
                   ):
                    best_losses = perm_losses

            # append to losses
            losses['sample'].extend(best_losses['sample'])
            losses['batch'].append(best_losses['batch'][0])

        losses['batch'] = torch.stack(losses['batch'], dim=0)
        return losses

    def forward(self, y_pred, y_true):
        raw = self.forward_no_reduction(y_pred, y_true)
        raw_batch = raw['batch']
        if self.reduction == 'none':
            pass
        elif self.reduction == 'mean':
            raw['batch'] = raw['batch'].mean()
        elif self.reduction == 'sum':
            raw['batch'] = raw['batch'].sum()
        return raw

register_loss_class('composite', CompositeLoss)
