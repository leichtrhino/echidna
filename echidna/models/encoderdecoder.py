
import typing as tp
import torch

class EncoderDecoderModel(torch.nn.Module):
    """
    """

    def __init__(self,
                 encoder_class : tp.Type[torch.nn.Module],
                 encoder_params : tp.Dict[str, object],
                 decoder_class : tp.Type[torch.nn.Module],
                 decoder_params : tp.Dict[str, object],
                 ):
        """
        """

        super().__init__()
        self.encoder = encoder_class(**encoder_params)
        self.decoder = decoder_class(**decoder_params)

    def forward(self, x):
        """
        Parameter
        ---------
        x : torch.Tensor
        """
        enc_out = self.encoder(x)
        if type(enc_out) == tuple or type(enc_out) == list:
            dec_out = self.decoder(*enc_out)
        elif type(enc_out) == dict:
            dec_out = self.decoder(**enc_out)
        else:
            dec_out = self.decoder(enc_out)
        return dec_out

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

    def forward_embd_feature(self) -> int:
        return None

    def forward_embd_dim(self) -> int:
        return None

    def forward_embd_length(self, l_in : int) -> int:
        return None

    def parameter_list(self, base_lr):
        return self.encoder.parameter_list(base_lr) \
            + self.decoder.parameter_list(base_lr) \


