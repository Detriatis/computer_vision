import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer 
from torch.nn.modules.loss import _Loss
from main.utils.cv_dataclasses import ImagesDataset
from main.utils.image_loading import load_directory_images
from main.utils.scikit_interface import ImageNormalizer, DescriptorExtractor, BoWTransformer, SKlearnPyTorchClassifier

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import multiprocessing as mp

mp.set_start_method('forkserver', force=True)

# CNN parameters
class Net(nn.Module):
    def __init__(self, convs: int = 1):
        super().__init__()
        self.convs = convs - 1 
        self.conv1 = nn.Conv2d(3, 64, 5, padding='same')
        self.conv1h = nn.Conv2d(64, 64, 5, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 128, 5, padding='same')
        self.conv2h = nn.Conv2d(128, 128, 5, padding='same')
        in_spatial = 96 // 2 // 2 
        self.fc1 = nn.Linear(128 * in_spatial * in_spatial, 120)
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

search_space_cnn = {
    'convs': Integer(0, 5),  
    'lr': Real(1e-4, 1e-1, prior='log-uniform'),
    'momentum': Real(0.0, 0.99),
    'epochs': Integer(5, 30),
    'batch_size': Categorical([32, 64, 128]) 
}

DIRECTORYPATH = Path("/home/michaeldodds/Projects/manchester/computer_vision/datasets/processed/")

if __name__ == '__main__':
    directories = ['stl10', 'cifar-10', 'mammals']
    for dir in directories: 
        imgs = load_directory_images(DIRECTORYPATH / dir, train=True)

        opt = BayesSearchCV(
            SKlearnPyTorchClassifier(Net),
            search_space_cnn,
            n_iter=32,
            scoring='accuracy',
            cv=5,
            verbose=1,
            random_state=0,
            n_jobs=1
            )
        
        opt.fit(np.array(imgs.imgs), np.array(imgs.labels) - 1)

        result_df = pd.DataFrame(opt.cv_results_)[[
        'params', 'mean_test_score', 'std_test_score'
        ]].sort_values('mean_test_score', ascending=False)        
        print(result_df)

        result_df['dataset'] = dir
                                    
        iterative_path = Path("/home/michaeldodds/Projects/manchester/computer_vision/results/hyperopt")
        iterative_path  = iterative_path / 'cnn_iterative_experiments.csv'
        if iterative_path.exists():
            df = pd.read_csv(iterative_path)
            iterative_df = pd.concat([result_df, df])
            iterative_df.to_csv(iterative_path, index=False)
        else:     
            result_df.to_csv(iterative_path, index=False)

