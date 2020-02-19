# -*- coding: utf-8 -*- 
# @Time : 2019-12-15 15:13 
# @Author : Trible 

import torch
from torch import nn

x = torch.randint(0, 9, (2, 3, 1))
# print(x)
xs = torch.randn(2, 3, 10)

# print(xs)
x1 = x.reshape(2, -1)
xs1 = xs.argmax(dim=2)
print(x1)
print(xs1)
print(torch.sum(x1 == xs1, dim=1))

# loss_fun = nn.CrossEntropyLoss()
# loss = loss_fun(xs.reshape(-1, 10), x.reshape(-1, ))
# print(loss)
