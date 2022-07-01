
import torch


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

