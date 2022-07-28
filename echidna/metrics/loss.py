
import torch

class Loss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f'invalid reduction "{reduction}"')
        self.reduction = reduction

    @property
    def domains(self):
        raise NotImplementedError()

    def to_dict(self):
        return {
            'type': _reverse_loss_map[type(self)],
            'args': self.to_dict_args(),
        }

    def to_dict_args(self):
        return {
            'reduction': self.reduction,
        }

    @classmethod
    def from_dict(cls, d : dict):
        loss_type = d['type']
        loss_class = _loss_map[loss_type]
        return loss_class.from_dict_args(d['args'])

    @classmethod
    def from_dict_args(cls, d : dict):
        return cls(reduction=d.get('reduction', 'mean'))

    def forward_no_reduction(self, y_pred, y_true):
        raise NotImplementedError()

    def forward(self, y_pred, y_true):
        raw = self.forward_no_reduction(y_pred, y_true)
        if self.reduction == 'none':
            return raw
        elif self.reduction == 'mean':
            return raw.mean()
        elif self.reduction == 'sum':
            return raw.sum()

_loss_map = {
    'loss': Loss,
}
_reverse_loss_map = dict((v, k) for k, v in _loss_map.items())

def register_loss_class(name, loss_class):
    _loss_map[name] = loss_class
    _reverse_loss_map[loss_class] = name

def get_loss_class(loss_name):
    if loss_name not in _loss_map:
        raise ValueError(f'{loss_name} is not known loss')
    return _loss_map[loss_name]

def get_loss_name(loss_class):
    if loss_class not in _reverse_loss_map:
        raise ValueError(f'{loss_class} is not known loss')
    return _reverse_loss_map[loss_class]
