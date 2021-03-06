# -*- coding: utf-8 -*-
"""fc.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vXS1MaSjWIIZrRJY1F63A0WhRQLarYzQ
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt

BATCH_SIZE = 32

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ]))
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ]))

class Net_web(torch.nn.Module):
    def __init__(self):
        super(Net_web, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

the_net = Net_web()
print(the_net)

optimizer = optim.SGD(the_net.parameters(), lr=0.01)

train_loader = Data.DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=True)

'''
Train the model by seeing whole trainning dataset once.
'''
def train(epoc):
    the_net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = the_net(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoc, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
'''
Compute average loss over the whole dataset.
'''
def get_train_loss():
  the_net.eval()
  train_loss = 0
  for batch_idx, (data, target) in enumerate(train_loader):
    output = the_net(data)
    train_loss += F.cross_entropy(output, target, reduction='sum')
  return train_loss/len(train_loader.dataset)

'''
Test the model over test dataset.
'''
def test():
  the_net.eval()
  test_loss = 0 # accumulated losses
  correct = 0 # number of correctly classified examples
  with torch.no_grad():
    for data, target in test_loader:
      output = the_net(data)
      test_loss += F.cross_entropy(output, target, reduction='sum').item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum().item()
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss / len(test_loader.dataset), correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

tol = 1e-3
loss_previous = 1e10
epoc = 1
train(epoc)
loss_next = get_train_loss()
while(loss_previous - loss_next > tol):
    epoc += 1
    train(epoc)
    loss_previous = loss_next
    loss_next = get_train_loss()
    print('Loss after previous epoc: {}'.format(loss_previous))
    print('Loss after current epoc: {}'.format(loss_next))

torch.save(the_net, 'mnist-fc')

the_net = None
the_net = torch.load('mnist-fc')
test()
