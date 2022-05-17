
from math import pi, sqrt, cos, sin
import torch

def init_conv_weight(conv : torch.nn.Conv1d) -> None:
    torch.nn.init.xavier_normal_(conv.weight)
    if conv.bias is not None:
        fan_out, fan_in = \
            torch.nn.init._calculate_fan_in_and_fan_out(conv.weight)
        if (fan_in + fan_in) != 0:
            std = sqrt(2 / (fan_in + fan_out))
            torch.nn.init.normal_(conv.bias, std=std)

def init_lstm_weight(lstm_layer : torch.nn.LSTM, hidden_size):
    N = hidden_size
    for name, param in lstm_layer.named_parameters():
        N = hidden_size
        if 'weight_ih' in name:
            for i, gain in enumerate([
                    torch.nn.init.calculate_gain('sigmoid'),
                    torch.nn.init.calculate_gain('sigmoid'),
                    torch.nn.init.calculate_gain('tanh'),
                    torch.nn.init.calculate_gain('sigmoid')
            ]):
                torch.nn.init.xavier_uniform_(param[i*N:(i+1)*N], gain)
        elif 'weight_hh' in name:
            for i, gain in enumerate([
                    torch.nn.init.calculate_gain('sigmoid'),
                    torch.nn.init.calculate_gain('sigmoid'),
                    torch.nn.init.calculate_gain('tanh'),
                    torch.nn.init.calculate_gain('sigmoid')
            ]):
                torch.nn.init.orthogonal_(param[i*N:(i+1)*N], gain)
        elif 'bias' in name:
            param.data.fill_(0)
            param.data[N:2*N].fill_(1) # for forget gate

def generate_dft_matrix(n_fft):
    phi = 2*pi*torch.arange(1, n_fft + 1, dtype=torch.float) / n_fft
    basis = torch.arange(n_fft // 2, dtype=torch.float).unsqueeze(-1)
    return torch.cat((torch.cos(phi*basis), torch.sin(phi*basis)))

