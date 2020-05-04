#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WDCGAN vs DCGAN
1. remove sigmoid in the last layer of discriminator(classification -> regression)                                       # 回归问题,而不是二分类概率
2. no log Loss (Wasserstein distance)
3. clip param norm to c (Wasserstein distance and Lipschitz continuity)
4. No momentum-based optimizer, use RMSProp，SGD instead
"""


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


image_size = 28 #图像尺寸大小
input_dim = 100 #输入给生成器的向量维度，维度越大可以增加生成器输出样本的多样性
num_channels = 1 # 图像的通道数
num_features = 64 #生成器中间的卷积核数量
batch_size = 64 #批次大小
num_epochs = 30
clamp_num = 0.01 # WGAN clip gradient
use_cuda = torch.cuda.is_available() #定义一个布尔型变量，标志当前的GPU是否可用


train_dataset = dsets.MNIST(root='./data',  #文件存放路径
                            train=True,   #提取训练集
                            transform=transforms.ToTensor(), 
                            download=True) 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)


# Define some useful functions
def weight_init(m):
    """模型参数初始化．
    默认的初始化参数卷积核的权重是均值大概为0，方差在10^{-2}. BatchNorm层的权重均值是大约0.5，方差在0.2左右
    使用如下初始化方式可以，可以让方差更小，使得收敛更快"""
    class_name=m.__class__.__name__
    if class_name.find('conv')!=-1:
        m.weight.data.normal_(0,0.02)
    if class_name.find('norm')!=-1:
        m.weight.data.normal_(1.0,0.02)
        
def make_show(img):
    """将张量变成可以显示的图像"""
    img = img.data.expand(batch_size, 3, image_size, image_size)
    return img

def imshow(inp, title=None, ax=None):
    """Imshow for Tensor."""
    if inp.size()[0] > 1:
        inp = inp.numpy().transpose((1, 2, 0))
    else:
        inp = inp[0].numpy()
    mvalue = np.amin(inp)
    maxvalue = np.amax(inp)
    if maxvalue > mvalue:
        inp = (inp - mvalue)/(maxvalue - mvalue)
    ax.imshow(inp)
    if title is not None:
        ax.set_title(title)
        
        
        
# Generater and Discriminator
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


class ModelD(nn.Module):
    def __init__(self):
        super(ModelD,self).__init__()
        self.model=nn.Sequential() #序列化模块构造的神经网络
        self.model.add_module('conv1',nn.Conv2d(num_channels, num_features, 5, 2, 0, bias=True)) #卷积层
        self.model.add_module('relu1',nn.LeakyReLU(0.2, inplace = True)) #激活函数使用了leakyReLu，可以防止dead ReLu的问题
        self.model.add_module('conv2',nn.Conv2d(num_features, num_features * 2, 5, 2, 0, bias=True))
        self.model.add_module('bnorm2',nn.BatchNorm2d(num_features * 2)) 
        self.model.add_module('fc1', nn.Linear(num_features * 2 * 4 * 4, num_features))
        self.model.add_module('relu2',nn.LeakyReLU(0.2, inplace = True))
        self.model.add_module('fc2', nn.Linear(num_features, 1)) 
        
    def forward(self,x):
        for name, module in self.model.named_children():
            if name == 'fc1':
                x = x.view(-1, num_features * 2 * 4 * 4)
            x = module(x)
        return x
    


# 构建模型，并加载到GPU上
netG = ModelG().cuda() if use_cuda else ModelG()
netG.apply(weight_init)

netD=ModelD().cuda() if use_cuda else ModelD()
netD.apply(weight_init)


# 要优化两个网络，所以需要有两个优化器
optimizerD = optim.RMSprop(netD.parameters(), lr=0.0001)
optimizerG = optim.RMSprop(netG.parameters(), lr=0.0001)


error_G = None #总误差
for epoch in tqdm(range(num_epochs)):
    for batch_idx, (data, target) in enumerate(train_loader):
        one = torch.ones((data.shape[0], 1))
        mone = -1 * one
        
        # clip param for discriminator
        for parm in netD.parameters():
                parm.data.clamp_(-clamp_num, clamp_num)
                
        # Train D
        optimizerD.zero_grad()
        data, target = data.clone().detach().requires_grad_(True), target.clone().detach()
        label = torch.ones(data.size()[0])  #正确的标签是1（真实）        
        if use_cuda:
            data, target, label = data.cuda(), target.cuda(), label.cuda()
        netD.train()
        output = netD(data) #放到辨别网络里辨别
        output.backward(one) #辨别器的反向误差传播
        
        noise = torch.Tensor(np.random.normal(0, 1, (data.size()[0], input_dim, 1, 1)))
        fake_pic = netG(noise).detach() #这里的detach是为了让生成器不参与梯度更新
        output2 = netD(fake_pic) #用辨别器识别假图像
        output2.backward(mone) #反向传播误差
                
        optimizerD.step() 
        
        # Train G
        if error_G is None or np.random.rand() < 0.5:
            optimizerG.zero_grad() #清空生成器梯度
            #注意生成器的目标函数是与辨别器的相反的，故而当辨别器无法辨别的时候为正确
            noise.data.normal_(0,1) #重新随机生成一个噪声向量
            netG.train()
            fake_pic = netG(noise) #生成器生成一张伪造图像
            output = netD(fake_pic) #辨别器进行分辨
            output.backward(one) #反向传播
            optimizerG.step() #优化网络
            


# 绘制一些样本
noise = torch.FloatTensor(batch_size, input_dim, 1, 1)
noise.data.normal_(0,1)
noise = noise.cuda() if use_cuda else noise
netG.eval()
fake_u = netG(noise)
fake_u = fake_u.cpu() if use_cuda else fake_u
noise = noise.cpu() if use_cuda else noise
img = fake_u #.expand(batch_size, 3, image_size, image_size) #将张量转化成可绘制的图像

f, axarr = plt.subplots(8,8, sharex=True, figsize=(15,15))
for i in range(batch_size):
    axarr[i // 8, i % 8].axis('off')
    imshow(img[i].data, None, axarr[i // 8, i % 8])
    


torch.save(netG.state_dict(), r'C:\\Users\\HP\\Desktop\\深度学习与强化学习\\BP to Sparseness\\netG_param.pkl')
