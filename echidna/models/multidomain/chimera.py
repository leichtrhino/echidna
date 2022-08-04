
import typing as tp
import torch

from ..utils import init_conv_weight, init_module

class EmbeddingHead(torch.nn.Module):
    """
    """

    def __init__(self,
                 in_feature : int,
                 embd_feature : int,
                 embd_dim : int,
                 ):
        """
        """

        super().__init__()

        self.conv = torch.nn.Conv1d(in_feature,
                                    embd_feature * embd_dim,
                                    1)
        init_conv_weight(self.conv)

        self.in_feature = in_feature
        self.embd_feature = embd_feature
        self.embd_dim = embd_dim

    def forward(self, x):
        """
        Parameter
        ---------
        x : torch.Tensor
        """
        embd_head = self.conv(x)\
                        .unflatten(-2, (self.embd_feature, self.embd_dim))\
                        .transpose(-1, -2)\
                        .sigmoid()
        embd_head = embd_head / embd_head.norm(dim=-1, keepdim=True)
        return embd_head

    def forward_feature_size(self) -> int:
        return self.embd_feature

    def forward_feature_dim(self) -> int:
        return self.embd_dim


class ChimeraNet(torch.nn.Module):
    """
    """

    def __init__(self,
                 encoder_class : tp.Type[torch.nn.Module],
                 decoder_class : tp.Type[torch.nn.Module],
                 hyperparameters : tp.Dict[str, object],
                 ):
        """
        """

        super().__init__()
        base_hyperparameters = hyperparameters['base']
        embd_hyperparameters = hyperparameters['embd']

        self.encoder = init_module(encoder_class, base_hyperparameters)
        self.decoder = init_module(decoder_class, base_hyperparameters)

        self.embedding_head = EmbeddingHead(
            self.encoder.forward_feature_size(),
            embd_hyperparameters['embd_feature'],
            embd_hyperparameters['embd_dim'],
        )

    def forward(self, x):
        """
        Parameter
        ---------
        x : torch.Tensor
        """
        embd = self.encoder(x)
        if type(embd) == tuple:
            if len(embd) > 1:
                embd, supplements = embd[0], embd[1:]
            else:
                embd, supplements = embd[0], None
        else:
            supplements = None

        if supplements is not None:
            waveform = self.decoder(embd, *supplements)
        else:
            waveform = self.decoder(embd)

        embd_head = self.embedding_head(embd)

        return {
            'waves': waveform,
            'embd': embd_head,
        }

    def forward_wave_length(self, l_in : int) -> int:
        """
        Parameter
        ---------
        l_in : int
        """
        return self.decoder.forward_length(
            self.encoder.forward_length(l_in))

    def reverse_wave_length(self, l_out : int) -> int:
        """
        Parameter
        ---------
        l_out : int
        """
        return self.encoder.reverse_length(
            self.decoder.reverse_length(l_out))

    def forward_wave_channel(self) -> int:
        return self.decoder.forward_feature_size()

    def forward_embd_feature(self) -> int:
        return self.embedding_head.forward_feature_size()

    def forward_embd_dim(self) -> int:
        return self.embedding_head.forward_feature_dim()

    def forward_embd_length(self, l_in : int) -> int:
        return self.encoder.forward_length(l_in)

    def parameter_list(self, base_lr):
        return self.encoder.parameter_list(base_lr) \
            + self.decoder.parameter_list(base_lr) \
            + [{'params': self.embedding_head.parameters()}]


