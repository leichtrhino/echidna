
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
            'reduction': self.reduction,
        }

    @classmethod
    def from_dict(cls, d : dict):
        return cls(d['reduction'])

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

