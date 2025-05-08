import numpy as np
import matplotlib.pyplot as plt
import cv2
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
from sklearn.model_selection import StratifiedKFold
from skopt.space import Real, Integer, Categorical
import multiprocessing as mp
import time

mp.set_start_method('forkserver', force=True)

# CNN parameters
class Net(nn.Module):
    def __init__(self, 
                 channels: int=3, 
                 kernel_width: int=5, 
                 class_heads: int = 10,
                 img_size: int | tuple[int, int] = 32,
                 ):
        
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 6, kernel_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 16, kernel_width), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

        )
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        with torch.no_grad():
            dummy = torch.zeros(1, channels, *img_size)
            n_flat = self.features(dummy).numel()
        
        self.fc1 = nn.Linear(n_flat, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, class_heads)
        
        print(channels, kernel_width, class_heads) 
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
       
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x 

class MultiKernelBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        k = out_ch // 3
        r = out_ch - (3 * k)
        self.branch3 = nn.Conv2d(in_ch, k, 3, padding=1, bias=False)
        self.branch5 = nn.Conv2d(in_ch, k + r, 5, padding=2, bias=False)
        self.branch7 = nn.Conv2d(in_ch, k, 7, padding=3, bias=False)
        self.bn      = nn.BatchNorm2d(out_ch)
        self.act     = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)
        return self.act(self.bn(x))

class NetSmaller(nn.Module):
    def __init__(self,
                 channels = 3,
                 class_heads: int = 10,
                 width: int=32,
                 **kwargs, 
                 ):
        
        super().__init__()

        w = lambda c: int(c * width / 32)

        self.stem = nn.Sequential(
            nn.Conv2d(channels, w(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w(32)),
            nn.SiLU(inplace=True),
        )
        
        self.stage1 = nn.Sequential(
            MultiKernelBlock(w(32), w(64)),
            nn.MaxPool2d(2)
        )  
        
        self.stage2 = nn.Sequential(
            MultiKernelBlock(w(64), w(128)),
            nn.MaxPool2d(2)
        )
        
        self.stage3 = nn.Sequential(
            MultiKernelBlock(w(128), w(256)),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(w(256), class_heads)
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.init.zero_s(m.bias) 
    
    def forward(self, x):

        x = self.stem(x) 
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)

        return x 



search_space_cnn = {
    'net__kernel_width': Categorical([3, 5, 7]),
    'net__class_heads': Categorical([20]),
    'net__channels': Categorical([3]),
    'optim__lr': Real(1e-4, 1e-1, prior='log-uniform'),
    'optim__momentum': Real(0.0, 0.99),
    'epochs': Integer(2, 30),
    'batch_size': Categorical([4, 8, 16]) 
}

search_space_norms = [
            cv2.NORM_MINMAX,
            ]

nets = [
    Net,
    NetSmaller,
]

DIRECTORYPATH = Path("/home/michaeldodds/Projects/manchester/computer_vision/datasets/processed/")
colours = [False, True]
image_resize = (96, 96)
directories = ['stl10', 'mammals']
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
iterations_total = (len(directories) * len(colours) * len(search_space_norms) * len(nets)) - 1

def ticker(res):
    i = len(res.x_iters)          # how many points tried so far
    print(f"Iteration {i}/{opt.n_iter}")

if __name__ == '__main__':
    i = 0
    for dir in directories:
        
        for colour in colours:
            if colour: 
                search_space_cnn['net__channels'] = Categorical([3])
                col = None 
                norm_tuple = (0.5, 0.5, 0.5)
            else: 
                search_space_cnn['net__channels'] = Categorical([1])
                col = cv2.COLOR_BGR2GRAY
                norm_tuple = (0.5)
            
            if dir == 'mammals':
                search_space_cnn['net__class_heads'] = Categorical([20])
            else: 
                search_space_cnn['net__class_heads'] = Categorical([10])
            
            imgs = load_directory_images(DIRECTORYPATH / dir, train=True, subsample=0)
            print(f'Training on {len(imgs.imgs)}') 
            
            
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(norm_tuple, norm_tuple),
                transforms.Resize(image_resize)]
            )
            
            for norm_type in search_space_norms:
                i += 1
            
                imagenormalizer = ImageNormalizer(norm_type=norm_type, alpha=0, beta=255, dtype=np.uint8, col=col).fit(imgs.imgs)
                norm_x = imagenormalizer.transform(imgs.imgs)
                if not colour:
                    norm_x = np.expand_dims(norm_x, -1)
                
                for net in nets:
                    if net is NetSmaller:
                        if not colour or norm_type == 'eq_hist': 
                            continue   
                    
                    
                    opt = BayesSearchCV(
                        SKlearnPyTorchClassifier(net, device='cuda', transform=transform, image_size = (96, 96)),
                        search_space_cnn,
                        n_iter=16,
                        scoring=None,
                        cv=cv,
                        verbose=1,
                        random_state=0,
                        n_jobs=1,
                        refit=True,
                        )

                    opt.fit(np.array(norm_x), np.array(imgs.labels) - 1, callback=[ticker])

                    result_df = pd.DataFrame(opt.cv_results_)[[
                    'params', 'mean_test_score', 'std_test_score'
                    ]].sort_values('mean_test_score', ascending=False)

                    print(opt.cv_results_)       

                    result_df['dataset'] = dir
                    result_df['norm_type'] = norm_type
                    if colour: 
                        result_df['colour'] = 'rgb'
                    else: 
                        result_df['colour'] = 'greyscale'
                    if  net is Net: 
                        result_df['classifier'] = 'classic_cnn'
                    if  net is NetSmaller:
                        result_df['classifier'] = 'split_cnn'

                    print(f'SAVING DATA iteration {i} of {iterations_total}')             
                    iterative_path = Path("/home/michaeldodds/Projects/manchester/computer_vision/results/hyperopt")
                    iterative_path  = iterative_path / 'cnn_iterative_experiments3.csv'
                    if iterative_path.exists():
                        df = pd.read_csv(iterative_path)
                        iterative_df = pd.concat([result_df, df])
                        iterative_df.to_csv(iterative_path, index=False)
                    else:     
                        result_df.to_csv(iterative_path, index=False)

                    if col: 
                        break 
