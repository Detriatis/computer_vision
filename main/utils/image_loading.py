import numpy as np 
import cv2 
import sys
from pathlib import Path 
from dataclasses import dataclass, field 
from main.utils.cv_dataclasses import Images

def load_image(filepath: Path, flag = cv2.IMREAD_COLOR_BGR):
    img = cv2.imread(filepath, flags = flag)
    return img 

def load_labeled_images(directorypath: Path) -> Images: 
    img_paths = list(directorypath.iterdir())
    imgs = [load_image(path) for path in img_paths]
    labels = [directorypath.name] * len(img_paths)  
    train = directorypath.parent.name == 'train'
    directory = directorypath.parent.parent.name 

    return Images(filepath=img_paths, 
                  labels=labels, 
                  train=train, 
                  directory=directory, 
                  imgs=imgs)

def load_directory_images(directorypath: Path) -> Images:
    labelled_dir_paths = list(directorypath.iterdir())
    all_img_list = [load_labeled_images(path) for path in labelled_dir_paths]
    all_imgs = Images.concat(all_img_list)
    print('Loaded images from:', all_imgs.directory, '\nTraining: ', all_imgs.train)
    return all_imgs
