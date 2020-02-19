# -*- coding: utf-8 -*- 
# @Time : 2019-12-15 20:08 
# @Author : Trible 

import torch
from torch import nn

class MyRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn_layer01 = nn.LSTM(60 * 3, 128, 2, batch_first=True)
        self.rnn_layer02 = nn.LSTM(32, 10, 2, batch_first=True)
        self.output_layer = nn.Linear(128, 2 * 64)
        self.hn_layer = nn.Linear(2 * 128, 2 * 10)
        self.cn_layer = nn.Linear(2 * 128, 2 * 10)
        self.out_layer = nn.Linear(10, 40)

    def forward(self, x):
        inputs = x.permute(0, 3, 1, 2)
        inputs = inputs.reshape(-1, 120, 60 * 3)

        h00 = torch.zeros(2, x.shape[0], 128).cuda()
        c00 = torch.zeros(2, x.shape[0], 128).cuda()
        outputs01, (hn0, cn0) = self.rnn_layer01(inputs, (h00, c00))

        outputs01 = self.output_layer(outputs01[:, -1, :]).reshape(x.shape[0], 4, -1)
        hn0 = hn0.permute(1, 0, 2).reshape(x.shape[0], 2 * 128)
        hn0 = self.hn_layer(hn0).reshape(-1, 2, 10).permute(1, 0, 2).cuda()
        cn0 = cn0.permute(1, 0, 2).reshape(x.shape[0], 2 * 128)
        cn0 = self.hn_layer(cn0).reshape(-1, 2, 10).permute(1, 0, 2).cuda()

        outputs, (hn1, cn1) = self.rnn_layer02(outputs01, (hn0.contiguous(), cn0.contiguous()))
        outputs = outputs[:, -1, :]
        outputs = self.out_layer(outputs)
        return outputs.reshape(-1, 4, 10)

if __name__ == "__main__":
    x = torch.randn(2, 3, 60, 120)
    myrnn = MyRNN()
    out = myrnn(x)
    print(out.shape)