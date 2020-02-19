# -*- coding: utf-8 -*- 
# @Time : 2019-12-15 12:45 
# @Author : Trible

import torch
from torch import nn

class MyRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn_layer01 = nn.LSTM(60 * 3, 128, 2, batch_first=True)
        self.rnn_layer02 = nn.LSTM(32, 10, 4, batch_first=True)
        self.output_layer = nn.Linear(10, 40)

    def forward(self, x):
        inputs = x.permute(0, 3, 1, 2)
        inputs = inputs.reshape(-1, 120, 60 * 3)

        h00 = torch.zeros(2, x.shape[0], 128).cuda()
        c00 = torch.zeros(2, x.shape[0], 128).cuda()
        outputs01, (hn0, cn0) = self.rnn_layer01(inputs, (h00, c00))
        outputs01 = outputs01[:, -1, :].reshape(-1, 4, 32)

        h01 = torch.zeros(4, x.shape[0], 10).cuda()
        c01 = torch.zeros(4, x.shape[0], 10).cuda()
        outputs, (hn1, cn1) = self.rnn_layer02(outputs01, (h01, c01))
        outputs = outputs[:, -1, :]
        outputs = self.output_layer(outputs)
        return outputs.reshape(-1, 4, 10)

if __name__ == "__main__":
    x = torch.randn(2, 3, 60, 120)
    myrnn = MyRNN()
    out = myrnn(x)
    print(out.shape)


