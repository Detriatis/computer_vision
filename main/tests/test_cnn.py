from main.utils.image_loading import load_directory_images
from main.utils.cv_dataclasses import ImagesDataset
from main.pipeline.cnn import train_cnn, predict_cnn

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

CIFAR10 = "/home/REDACTED/Projects/manchester/computer_vision/datasets/processed/cifar-10/"

trainimgs = load_directory_images(CIFAR10, True)
testimgs = load_directory_images(CIFAR10, False)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = ImagesDataset(trainimgs, transform)
testset = ImagesDataset(testimgs, transform)
batch_size = 5

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                         shuffle=True, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    #npimg = img.numpy()
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.show()


## get some random training images
#dataiter = iter(trainloader)
#images, labels = next(dataiter)
## show images
#print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
#imshow(torchvision.utils.make_grid(images))
## print labels


class Net(nn.Module):
    def __init__(self, convs):
        super().__init__()
        self.convs = convs - 1 
        self.conv1 = nn.Conv2d(3, 64, 5, padding='same')
        self.conv1h = nn.Conv2d(64, 64, 5, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 128, 5, padding='same')
        self.conv2h = nn.Conv2d(128, 128, 5, padding='same')
        
        self.fc1 = nn.Linear(128 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        for _ in range(self.convs):
            x = F.relu(self.conv1h(x))
        x = self.pool(x) 
        x = F.relu(self.conv2(x))
        for _ in range(self.convs): 
            x = F.relu(self.conv2h(x))
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x) 
        return x

net = Net(3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

train_cnn(net, trainloader, criterion, optimizer, 2, None)
results = predict_cnn(net, testloader, classes, None)
print(results[0], results[1])

for classname, correct_count in results[0].items():
    accuracy = 100 * float(correct_count) / results[1][classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')