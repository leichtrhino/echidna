
import torch

class ToyEncoder(torch.nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.out_channel = out_channel

    def forward(self, x):
        return x[..., ::2]

    def forward_length(self, l_in):
        return l_in // 2

    def reverse_length(self, l_out):
        return l_out * 2

    def forward_channel(self):
        return self.out_channel

class ToyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat((x, x), dim=-1)

    def forward_length(self, l_in):
        return l_in * 2

    def reverse_length(self, l_out):
        return l_out // 2
