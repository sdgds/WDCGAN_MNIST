#!/usr/bin/env python
# encoding: utf-8

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Combinate generator and identificator to a new model
image_size = 28 
input_dim = 100 
num_channels = 1
num_features = 64 
batch_size = 64
class ModelG(nn.Module):
    def __init__(self):
        super(ModelG,self).__init__()
        self.model=nn.Sequential() #model为一个内嵌的序列化的神经网络模型    
        # 利用add_module增加层，nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0,  output_padding=0, groups=1, bias=True, dilation=1)
        # 输入图像大小为1，输出图像大小为W'=(W-1)S-2P+K+P'=(1-1)*2-2*0+5+0=5, 5*5
        self.model.add_module('deconv1',nn.ConvTranspose2d(input_dim, num_features*2, 5, 2, 0, bias=True))
        self.model.add_module('bnorm1',nn.BatchNorm2d(num_features*2))
        self.model.add_module('relu1',nn.ReLU(True))
        # 输入图像大小为5，输出图像大小为W'=(W-1)S-2P+K+P'=(5-1)*2-2*0+5+0=13, 13*13
        self.model.add_module('deconv2',nn.ConvTranspose2d(num_features*2, num_features, 5, 2, 0, bias=True))
        self.model.add_module('bnorm2',nn.BatchNorm2d(num_features))
        self.model.add_module('relu2',nn.ReLU(True))
        # 输入图像大小为13，输出图像大小为W'=(W-1)S-2P+K+P'=(13-1)*2-2*0+4+0=28, 28*28
        self.model.add_module('deconv3',nn.ConvTranspose2d(num_features, num_channels, 4, 2, 0,bias=True))
        self.model.add_module('sigmoid',nn.Sigmoid())
        
    def forward(self,x):
        """输出一张28*28的图像"""
        for name, module in self.model.named_children():
            x = module(x)
        return(x)
net_generator = ModelG()
net_generator.load_state_dict(torch.load('netG_param.pkl'))

class DCNN(nn.Module):
    def __init__(self):
        super(DCNN,self).__init__()        
        self.layer1 = nn.Sequential(
                nn.Conv2d(1,16,kernel_size=3), # 16, 26 ,26
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True))        
        self.layer2 = nn.Sequential(
                nn.Conv2d(16,32,kernel_size=3),# 32, 24, 24
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=2)) # 32, 12,12     (24-2) /2 +1
        self.layer3 = nn.Sequential(
                nn.Conv2d(32,64,kernel_size=3), # 64,10,10
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
                nn.Conv2d(64,128,kernel_size=3),  # 128,8,8
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=2))  # 128, 4,4
        self.fc = nn.Sequential(
                nn.Linear(128 * 4 * 4,1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024,128),
                nn.ReLU(inplace=True),
                nn.Linear(128,10))
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)        
        return x
net_identificator = DCNN() 
net_identificator.load_state_dict(torch.load('netI_param.pkl'))


class Model(nn.Module):
    def forward(self, x):
        x = net_generator(x)
        x = net_identificator.layer1(x)
        x = net_identificator.layer2(x)  
        x = net_identificator.layer3(x)  
        x = net_identificator.layer4(x)  
        x = x.view(x.size(0),-1)
        x = net_identificator.fc(x)
        return x


# From category of DCNN to vector of DCGAN (similar to AM algorithm)
def category2vector(model, category):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    vector = torch.zeros((1,100,1,1))
    step = 0
    count = 0
    while count < 5:
        step += 1
        random_vectors = torch.Tensor(np.random.normal(0,1,(1,100,1,1)))
        output = model.forward(random_vectors)
        if torch.topk(output, 1)[1].item()>0 and torch.topk(output, 1)[1].item()==category:
            count += 1
            print(count)
            vector += random_vectors * torch.topk(output, 1)[1].item()
        if step == 1000:
            random_vectors = torch.Tensor(np.random.normal(0,1,(1,100,1,1))).requires_grad_(True)
            optimizer = optim.Adam([random_vectors], lr=0.1, weight_decay=1e-6)
            for i in tqdm(range(500)):
                output = model.forward(random_vectors)
                loss = -output[0, category]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
            vector = random_vectors
            break
    return vector


# Visulization the generated picture
model = Model()
category_vectors = category2vector(model, 0)
img = net_generator(category_vectors)
plt.figure()
plt.imshow(img[0,0,:].data.numpy())
plt.axis('off')

