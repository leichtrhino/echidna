
import torch

from .loss import Loss, register_loss_class

class _EmbeddingLoss(Loss):
    def __init__(self,
                 mode,
                 label='argmax', # or softmax
                 weight='none', # or magnitude_ratio
                 reduction='mean'):
        super().__init__(reduction)
        if weight not in ('none', 'magnitude_ratio'):
            raise ValueError(f'invalid weight type {weight}')
        if label not in ('argmax', 'softmax'):
            raise ValueError(f'invalid label type {label}')

        self.label = label
        self.weight = weight
        self.mode = mode

    @property
    def domains(self):
        return ('embd',)

    def to_dict_args(self):
        return {
            'label': self.label,
            'weight': self.weight,
            'reduction': self.reduction,
        }

    @classmethod
    def from_dict_args(cls, d : dict):
        return cls(
            label=d.get('label', 'argmax'),
            weight=d.get('weight', 'none'),
            reduction=d.get('reduction', 'mean'),
        )

    def forward_no_reduction(self,
                             e_pred : torch.Tensor,
                             e_true : torch.Tensor):
        if self.label == 'argmax':
            C = e_pred.shape[-1]
            device = e_true.device
            label = torch.eye(C, device=device)[e_true.argmax(dim=-1)]
        elif self.label == 'softmax':
            label = e_true.softmax(dim=-1)

        weight = None
        if self.weight == 'magnitude_ratio':
            weight = e_true.abs().sum(dim=-1, keepdims=True) \
                / e_true.abs().sum().clamp(min=1e-3)

        raw = deep_clustering_loss(e_pred, label, weight, self.mode)
        return raw

class DeepClusteringLoss(_EmbeddingLoss):
    def __init__(self, label='argmax', weight='none', reduction='mean'):
        super().__init__('deep_clustering', label, weight, reduction)

class DeepLDALoss(_EmbeddingLoss):
    def __init__(self, label='argmax', weight='none', reduction='mean'):
        super().__init__('deep_lda', label, weight, reduction)

class GraphLaplacianLoss(_EmbeddingLoss):
    def __init__(self, label='argmax', weight='none', reduction='mean'):
        super().__init__('graph_laplacian_distance', label, weight, reduction)

class WhitenedKMeansLoss(_EmbeddingLoss):
    def __init__(self, label='argmax', weight='none', reduction='mean'):
        super().__init__('whitened_kmeans', label, weight, reduction)

register_loss_class('deep_clustering', DeepClusteringLoss)
register_loss_class('deep_lda', DeepLDALoss)
register_loss_class('graph_laplacian', GraphLaplacianLoss)
register_loss_class('whitened_kmeans', WhitenedKMeansLoss)


def deep_clustering_loss(embd : torch.Tensor,
                         label : torch.Tensor,
                         weight : torch.Tensor=None,
                         mode : str='deep_clustering'):
    """
    loss functions for deep clustering
    embd: (batch_size, *, embd_dim)
    label: (batch_size, *, label_dim)
    weight: (batch_size, *)
    """

    if not mode in ('deep_clustering',
                    'deep_lda',
                    'graph_laplacian_distance',
                    'whitened_kmeans'):
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

    if mode == 'deep_clustering':
        return torch.sum(embd.transpose(1, 2).bmm(embd) ** 2, dim=(1, 2)) \
            + torch.sum(label.transpose(1, 2).bmm(label) ** 2, dim=(1, 2)) \
            - 2 * torch.sum(embd.transpose(1, 2).bmm(label) ** 2, dim=(1, 2))

    elif mode == 'deep_lda':
        YtV = label.transpose(1, 2).bmm(embd)
        YtY = label.transpose(1, 2).bmm(label) \
            + torch.eye(C, device=label.device)
        YtYu = torch.linalg.cholesky(YtY)
        YtYinv = YtYu.cholesky_inverse()
        return torch.sum((embd - label.bmm(YtYinv.bmm(YtV))) ** 2, dim=(1, 2)) \
            / torch.sum((embd-embd.mean(dim=-2, keepdims=True))**2, dim=(1, 2))

    elif mode == 'graph_laplacian_distance':
        D_V = torch.bmm(
            embd,
            torch.sum(embd.transpose(1, 2), dim=2, keepdims=True)
        ).clamp(min=1e-24)
        D_Y = torch.bmm(
            label,
            torch.sum(label.transpose(1, 2), dim=2, keepdims=True)
        ).clamp(min=1e-24)
        return torch.sum(embd.transpose(1, 2).bmm(1/D_V*embd)**2, dim=(1, 2)) \
            + C \
            - 2 * torch.sum(
                torch.bmm(
                    (embd * torch.sqrt(D_V) ** -1).transpose(1, 2),
                    torch.sqrt(D_Y) ** -1 * label
                ) ** 2,
                dim=(1, 2)
            )

    elif mode == 'whitened_kmeans':
        VtV = embd.transpose(1, 2).bmm(embd) \
            + torch.eye(D, device=embd.device)
        VtVu = torch.linalg.cholesky(VtV)
        VtY = embd.transpose(1, 2).bmm(label)
        YtY = label.transpose(1, 2).bmm(label) \
            + torch.eye(C, device=label.device)
        YtYu = torch.linalg.cholesky(YtY)
        return D \
            - torch.diagonal(
                VtVu.cholesky_inverse()
                .bmm(VtY)
                .bmm(YtYu.cholesky_inverse())
                .bmm(VtY.transpose(1, 2)),
                dim1=-1,
                dim2=-2,
            ).sum(dim=-1)

