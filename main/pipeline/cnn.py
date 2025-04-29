import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
import torchvision.transforms as transforms 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer 
from torch.nn.modules.loss import _Loss
from main.utils.cv_dataclasses import ImagesDataset

def train_cnn(net: nn.Module, trainloader: torch.utils.data.DataLoader, criterion: _Loss, optimizer: Optimizer, steps: int, savepath = None):
    for epoch in range(steps):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999: 
                print(f'{epoch + 1}, {i + 1}, loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Finished Training')
    if savepath:
        torch.save(net.state_dict(), savepath)

def predict_cnn(net, testloader: torch.utils.data.DataLoader, classes, savepath = None):
 
    if savepath: 
        net.load_state_dict(torch.load(savepath, weights_only=True))
    if not net:
        raise TypeError('Net or Savepath must be specified')
    
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
        return correct_pred, total_pred 
