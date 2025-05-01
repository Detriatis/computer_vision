import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

from main.utils.image_loading import  load_directory_images
from main.utils.cv2_interface import get_cv2_detector
from main.pipeline.feature_detection import calc_features_desc
from main.pipeline.clustering import bag_of_words 
from main.pipeline.classifier import fit_model, pred_model, model_accuracy

import torch
import torchvision.transforms as transforms 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

from main.utils.cv_dataclasses import ImagesDataset
from main.pipeline.cnn import train_cnn, predict_cnn

# Shared Parameters
DIRECTORYPATH = "/home/michaeldodds/Projects/manchester/computer_vision/datasets/processed/stl10" 

# Classic CV Parameters
CV_PARAMETERS = {
                'normaliser': {
                    'alpha': 0, 'beta': 255, 'norm_type': cv2.NORM_MINMAX
                    },

                'detector': {
                    'algo': 'SIFT',
                    'parameters':{ 
                        'nfeatures': 10000
                    }
                    },

                'features': {
                    'algo': 'SIFT'
                    },

                'bag_of_words': {
                    'k': 50, 'max_iters': 1000, 'subsamples': 10000
                    },
                'classifier': {
                    'algo': '',
                    'parameters': {
                    },
                }

                } 

normaliser = cv2.normalize
detection_algo = get_cv2_detector(CV_PARAMETERS['detector']['algo'])
feature_algo = get_cv2_detector(CV_PARAMETERS['features']['algo'])
loss_metric = None
clf = None 

# CNN parameters
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

criterion = nn.CrossEntropyLoss()
net = Net(3)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

if __name__ == '__main__':
    normaliser  = cv2.normalize    
    keypoint_detector = detection_algo.create(**CV_PARAMETERS['detector']['parameters'])
    descriptor = feature_algo.create()
    
    imgs = load_directory_images(DIRECTORYPATH, train = True) 
    t_imgs = load_directory_images(DIRECTORYPATH, train = False)

    # Run classical CV  
    p_imgs = calc_features_desc(imgs, normaliser, CV_PARAMETERS['normaliser'], keypoint_detector, descriptor)
    tp_imgs = calc_features_desc(t_imgs, normaliser, CV_PARAMETERS['normaliser'], keypoint_detector, descriptor)
    
    bow, means = bag_of_words(p_imgs, **CV_PARAMETERS['bag_of_words'])
    t_bow, _ = bag_of_words(tp_imgs, **CV_PARAMETERS['bag_of_words'], means=means)
    
    fit_model(clf, bow, imgs.labels)
    preds = pred_model(clf, t_bow)
    results = model_accuracy(loss_metric, preds, t_imgs.labels) 
    
    # Run CNN  
    transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
                )

    trainset = ImagesDataset(imgs, transform)
    testset = ImagesDataset(t_imgs, transform)
    batch_size = 5

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=1)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                            shuffle=True, num_workers=1)
    
    #TODO load class labels from the file in the directory
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    

    train_cnn(net, trainloader, criterion, optimizer, 2, None)
    
    results = predict_cnn(net, testloader, classes, None)

    for classname, correct_count in results[0].items():
        accuracy = 100 * float(correct_count) / results[1][classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')