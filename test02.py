# -*- coding: utf-8 -*- 
# @Time : 2019-12-16 9:21 
# @Author : Trible 

import torch
from PIL import Image
import os
import random
from torchvision import transforms
from MyRNN02 import MyRNN
import matplotlib.pyplot as plt

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_img = os.listdir("test_data")
img_sample = random.choice(test_img)
# img = Image.open(os.path.join("test_data", img_sample))
img = Image.open(r"C:\Users\Administrator\Desktop\3844.jpg")
img = img.resize((120, 60), 1)
img_data = data_transforms(img)
x = img_data.unsqueeze(0)
y = img_sample.split(".")[0]

net = MyRNN().cuda()
net.load_state_dict(torch.load("model/model06.pth"))
ys = net(x.cuda()).squeeze()
score = torch.mean(torch.max(torch.softmax(ys, dim=1), dim=1).values).item()
print("实际值：", y)
print("预测值：", ys.argmax(1).tolist())
print("flag: ", [int(i) for i in y] == ys.argmax(1).tolist())
print("置信度：", score)
img.show()
