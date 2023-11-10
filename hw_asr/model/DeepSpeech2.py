import torch
from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel
from hw_asr.model.RNN import RNN


class DeepSpeech2(BaseModel):

    def _get_output_h(self, h_input, params, is_weight):
        return (h_input + 2 * params['padding'][is_weight] -
                params['dilation'][is_weight] *
                (params['kernel_size'][is_weight] - 1) - 1) // params['stride'][is_weight] + 1

    def _calculate_out_size(self, size_input, is_weight: bool = False):
        for params in self.params_for_convolutions:
            size_input = self._get_output_h(size_input, params, is_weight)
        return size_input

    def __init__(self, n_feats, n_class, params_for_convolutions, params_for_rnn, fc_hidden=512):
        super().__init__(n_feats, n_class)
        self.params_for_convolutions = params_for_convolutions['convolutions']

        if not params_for_convolutions['batch_norm']:
            self.convolution_layers = nn.ModuleList([
                nn.Conv2d(
                    in_channels=params_for_convolutions['in_channels']
                    if ind == 0 else params_for_convolutions['convolutions'][ind - 1]['out_channels'],
                    out_channels=params['out_channels'],
                    kernel_size=params['kernel_size'],
                    dilation=params['dilation'],
                    stride=params['stride'],
                    bias=False,
                    padding=params['padding']
                ) for ind, params in enumerate(params_for_convolutions['convolutions'])
            ])
        else:
            self.convolution_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=params_for_convolutions['in_channels']
                        if ind == 0 else params_for_convolutions['convolutions'][ind - 1]['out_channels'],
                        out_channels=params['out_channels'],
                        kernel_size=params['kernel_size'],
                        dilation=params['dilation'],
                        stride=params['stride'],
                        bias=False,
                        padding=params['padding']
                    ),
                    nn.BatchNorm2d(params['out_channels']),
                    nn.Hardtanh()
                ) for ind, params in enumerate(params_for_convolutions['convolutions'])
            ])

        self.after_conv_height = self._calculate_out_size(n_feats)

        self.rnn_layers = nn.ModuleList(
            [RNN(input_size=(params_for_rnn['hidden_size']) * (1 + params_for_rnn['bidirectional']) if num
            else self.after_conv_height,
                 hidden_size=params_for_rnn['hidden_size'],
                 rnn_model=params_for_rnn['rnn_model'],
                 dropout=params_for_rnn['dropout'],
                 bidirectional=params_for_rnn['bidirectional'],
                 batch_norm=params_for_rnn['batch_norm']) for num in range(params_for_rnn['count'])]
        )
        self.is_norm = params_for_rnn['batch_norm']
        self.batch_norm = nn.BatchNorm1d(num_features=2 * params_for_rnn['hidden_size'])
        self.final_stage = nn.Linear(in_features=2 * params_for_rnn['hidden_size'], out_features=n_class)

    def forward(self, spectrogram: torch.Tensor, **batch):
        # print("BEFORE ALL", spectrogram.shape)
        spectrogram = spectrogram.unsqueeze(1)
        for convolution in self.convolution_layers:
            spectrogram = convolution(spectrogram)
        spectrogram = spectrogram.squeeze()
        # print("After CNN", spectrogram.shape)
        for rnn in self.rnn_layers:
            spectrogram = rnn(spectrogram, self.transform_input_lengths(batch['spectrogram_length']))

        if self.is_norm:
            spectrogram = nn.functional.leaky_relu(self.batch_norm(spectrogram))
        # print("BEFOR FINAL:", spectrogram.shape)
        return {"logits": self.final_stage(spectrogram.transpose(1, 2))}

    def transform_input_lengths(self, input_lengths):
        return self._calculate_out_size(input_lengths, is_weight=True)
