import numpy as np 
import cv2 
import sys
from pathlib import Path 
from dataclasses import dataclass, field 
from main.utils.cv_dataclasses import Images

def load_image(filepath: Path, flag = cv2.IMREAD_COLOR_BGR):
    img = cv2.imread(filepath, flags = flag)
    return img 

def load_labeled_images(directorypath: Path, train: bool = True, subsample: int = 0) -> Images:
    
    img_paths = list(directorypath.iterdir())
    if subsample: 
        sub_img_idx = np.random.choice(len(img_paths), size=subsample, replace=False)
        img_paths = [img_paths[i] for i in sub_img_idx] 
    imgs = [load_image(path) for path in img_paths]
    labels = [int(directorypath.name)] * len(img_paths)
    
    return Images(filepath=img_paths, 
                  labels=labels, 
                  train=train, 
                  directory=directorypath.parent, 
                  imgs=imgs)

def load_directory_images(directorypath: Path, train: bool = True, subsample: int = 0) -> Images:
    if isinstance(directorypath, str):
        directorypath = Path(directorypath)
    
    if train:
        directory = directorypath / 'train'
    else:  
        directory = directorypath / 'test'

    labelled_dir_paths = list(directory.iterdir())
    all_img_list = [load_labeled_images(path, train, subsample) for path in labelled_dir_paths]
    all_imgs = Images.concat(all_img_list)
    print('Loaded images from:', all_imgs.directory, '\nTraining: ', all_imgs.train)
    return all_imgs
