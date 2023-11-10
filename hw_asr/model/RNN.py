import torch
from torch import nn


class RNN(nn.Module):
    _torch_rnn = {
        'RNN': nn.RNN,
        'LSTM': nn.LSTM,
        'GRU': nn.GRU
    }
    _torch_unlinear = {
        'ReLU': nn.ReLU,
        'ReLU6': nn.ReLU6,
        'LeakyReLU': nn.LeakyReLU
    }

    def __init__(self, input_size: int,
                 hidden_size: int = 256,
                 rnn_model: str = 'GRU',
                 unlinear: str = 'LeakyReLU',
                 dropout: float = 0.05,
                 bidirectional: bool = True,
                 batch_norm: bool = True):
        super(RNN, self).__init__()
        self.RNN = self._torch_rnn[rnn_model](input_size=input_size, hidden_size=hidden_size,
                                              dropout=dropout, bidirectional=bidirectional,
                                              batch_first=True)
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.is_norm = batch_norm
        self.unlinear = self._torch_unlinear[unlinear]

    def forward(self, x: torch.Tensor, length_rnn: torch.Tensor):
        # x: (batch, feature, time)
        total_length = x.size(2)
        if self.is_norm:
            x = nn.functional.leaky_relu(self.batch_norm(x))
        ans = torch.nn.utils.rnn.pack_padded_sequence(x.transpose(1, 2),
                                                      batch_first=True,
                                                      lengths=length_rnn,
                                                      enforce_sorted=False)
        ans = self.RNN(ans)[0]
        ans = torch.nn.utils.rnn.pad_packed_sequence(ans, batch_first=True, total_length=total_length)[0]
        return ans.transpose(1, 2)
