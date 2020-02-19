# -*- coding: utf-8 -*- 
# @Time : 2019-12-15 16:13 
# @Author : Trible 

import torch
from torch import nn

class MyRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn_layer01 = nn.LSTM(60 * 3, 128, 8, batch_first=True)
        self.output_layer = nn.Sequential(
            nn.Linear(128, 256),
            nn.PReLU(),
            nn.Linear(256, 40)
        )

    def forward(self, x):
        inputs = x.permute(0, 3, 1, 2)
        inputs = inputs.reshape(-1, 120, 60 * 3)

        h00 = torch.zeros(8, x.shape[0], 128).cuda()
        c00 = torch.zeros(8, x.shape[0], 128).cuda()
        outputs, (hn0, cn0) = self.rnn_layer01(inputs, (h00, c00))
        outputs = outputs[:, -1, :]

        output = self.output_layer(outputs)

        return output.reshape(-1, 4, 10)