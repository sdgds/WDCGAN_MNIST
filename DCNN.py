#!/usr/bin/env python
# encoding: utf-8

from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


train_set = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
train_loader = dataloader.DataLoader(
    dataset=train_set,
    batch_size=100,
    shuffle=True,
)

test_set = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
test_loader = dataloader.DataLoader(
    dataset=test_set,
    batch_size=100,
    shuffle=False,
)


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


def Train(model, epoches, lr, train_data, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5,0.999))
    
    train_loss = []
    test_acc = []
    for epoch in range(epoches):
        print('Now epoch is %s' %epoch)
        l=[]
        for images, labels in tqdm(train_data):
            output = model(images)
            loss = criterion(output, labels)
            l.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
        train_loss.append(0.1*sum(l)/len(l))
        
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_data:
                output = model(images)
                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_acc.append(correct/total)
            
    return train_loss, test_acc
    
         
            
epoches = 30
lr = 0.001
model = DCNN()           
train_loss, test_acc = Train(model, epoches, lr, train_loader, test_loader)         
            
plt.figure()
plt.plot(train_loss)
plt.title('Train loss')

plt.figure()
plt.plot(test_acc)   
plt.title('Test acc')  
    


torch.save(model.state_dict(), r'C:\\Users\\HP\\Desktop\\深度学习与强化学习\\BP to Sparseness\\netI_param.pkl')
