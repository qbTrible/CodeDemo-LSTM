# -*- coding: utf-8 -*- 
# @Time : 2019-12-10 19:22 
# @Author : Trible 

import torch
import torch.optim as optim
import torch.utils.data as data
from torch import nn
from MyRNN02 import MyRNN
from My_data import MyDataset
import os


def train():
    net = MyRNN().cuda()
    if os.path.exists("model/model06.pth"):
        net.load_state_dict(torch.load("model/model06.pth"))
    loss_func = nn.CrossEntropyLoss().cuda()
    optmizer = optim.Adam(net.parameters())
    Batch_Size = 512

    mydata = MyDataset("data01")
    test_data = MyDataset("test_data")
    data_loader = data.DataLoader(mydata, batch_size=Batch_Size, shuffle=True, num_workers=8)
    test_loader = data.DataLoader(test_data, batch_size=Batch_Size, shuffle=True, num_workers=8)

    count = 0
    while True:
        # train()
        net.train()
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0
        for i, (x, y) in enumerate(data_loader):
            x, y = x.cuda(), y.cuda()
            ys = net(x)

            # value = torch.argmax(ys, dim=1)
            loss = loss_func(ys.reshape(-1, 10), y.reshape(-1, ))
            train_correct += torch.sum(torch.sum(ys.argmax(dim=2) == y.reshape(-1, 4), dim=1) == 4).float()
            train_total += len(y)
            optmizer.zero_grad()
            loss.backward()
            optmizer.step()
            if i % 10 == 0:
                print("epoch:", count, "batch:", i)
                print("loss:", loss.item())
                torch.save(net.state_dict(), "model/model06.pth")
            del x, y, ys, loss

        with torch.no_grad():
            net.eval()
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                ys = net(x)
                test_correct += torch.sum(torch.sum(ys.argmax(dim=2) == y.reshape(-1, 4), dim=1) == 4).float()
                test_total += len(y)
                del x, y, ys

        count += 1
        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        print("Train Accuracy: ", train_acc.item(), "| Test Accuracy: ", test_acc.item())
        del test_correct, test_total, test_acc
        del train_correct, train_total, train_acc


if __name__ == "__main__":
    train()
